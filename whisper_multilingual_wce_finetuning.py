import argparse
import random
from typing import Any, Dict, List, Union
from dataclasses import dataclass

import torch
from torch.nn import CrossEntropyLoss

from audiomentations import (
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PitchShift,
    TimeStretch,
)

from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import evaluate


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your script')

    parser.add_argument("--base_model_dir", type=str,
                        required=True, help="Base model directory.")
    parser.add_argument("--output_dir", type=str,
                        required=True, help="Output directory.")
    parser.add_argument("--dataset_dir", type=str,
                        required=True, help="Dataset directory.")
    parser.add_argument("--load_dataset_from_disk", type=bool,
                        required=True, help="Load dataset from disk.")
    parser.add_argument("--save_dataset_to_disk", type=bool,
                        required=True, help="Load dataset from disk.")
    parser.add_argument("--prune_well_datasets", type=bool,
                        required=True, help="Prune well datasets.")
    parser.add_argument("--augment_gl_data", type=bool,
                        required=True, help="Augment GL data.")
    parser.add_argument("--max_input_length", type=int,
                        required=True, help="Max input length.")
    parser.add_argument("--min_input_length", type=int,
                        required=True, help="Min input length.")
    parser.add_argument("--per_device_train_batch_size",
                        type=int, required=True, help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        required=True, help="Gradient accumulation step.")
    parser.add_argument("--learning_rate", type=float,
                        required=True, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float,
                        required=True, help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int,
                        required=True, help="Warmup steps.")
    parser.add_argument("--max_steps", type=int,
                        required=True, help="Max steps.")
    parser.add_argument("--gradient_checkpointing", type=bool,
                        required=True, help="Gradient checkpointing.")
    parser.add_argument("--fp16", type=bool, required=True, help="Use FP16.")
    parser.add_argument("--evaluation_strategy", type=str,
                        required=True, help="Evaluation strategy")
    parser.add_argument("--per_device_eval_batch_size", type=int,
                        required=True, help="Batch size per device on evaluation.")
    parser.add_argument("--predict_with_generate", type=bool,
                        required=True, help="Predice with generate.")
    parser.add_argument("--generation_max_length", type=int,
                        required=True, help="Generation max length.")
    parser.add_argument("--save_steps", type=int,
                        required=True, help="Save steps.")
    parser.add_argument("--eval_steps", type=int,
                        required=True, help="Evaluation steps.")
    parser.add_argument("--alpha_initial", type=float,
                        required=False, help="Initial value of alpha.")
    parser.add_argument("--alpha_final", type=float,
                        required=False, help="Final value of alpha.")
    parser.add_argument("--num_min_steps", type=int, required=False,
                        help="Minimum number of steps for WS-FT-LP-WCE.")
    parser.add_argument("--num_total_steps", type=int, required=False,
                        help="Max number of steps for WS-FT-LP-WCE.")
    parser.add_argument("--linear_progressive_weight", type=bool,
                        required=False, help="True for WS-FT-LP-WCE.")
    parser.add_argument("--alpha", type=float, required=False,
                        help="Value of alpha for WS-FT-DA-WCE.")
    parser.add_argument("--dynamic_weight_adaptation", type=bool,
                        required=False, help="True for WS-FT-DA-WCE.")

    args = parser.parse_args()

    return args


def main():

    args = parse_arguments()

    common_voice = DatasetDict()
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model_dir)

    max_label_length = model.config.max_length
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        args.base_model_dir)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.base_model_dir, task="transcribe")
    processor = WhisperProcessor.from_pretrained(
        args.base_model_dir, task="transcribe")

    if args.load_dataset_from_disk:
        common_voice = common_voice.load_from_disk(args.dataset_dir)
    else:
        if args.augment_gl_data:
            augmentation = Compose(
                [
                    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2,
                                leave_length_unchanged=False),
                    Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.1),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
                    OneOf(
                        [
                            AddGaussianNoise(min_amplitude=0.005,
                                             max_amplitude=0.015, p=1.0),
                        ],
                        p=0.2,
                    ),
                ]
            )

        def remove_unnecessary_spaces(batch):
            batch['sentence'] = batch['sentence'].replace("¿ ", "¿").replace(" ?", "?").replace("¡ ", "¡").replace(
                " !", "!").replace(" .", ".").replace(" ,", ",").replace(" :", ":").replace(" ;", ";")
            return batch

        def augment_dataset(batch):
            # load and (possibly) resample audio data to 16kHz
            sample = batch["audio"]

            # apply augmentation
            augmented_waveform = augmentation(
                sample["array"], sample_rate=sample["sampling_rate"])

            batch["audio"]["array"] = augmented_waveform

            return batch

        def prepare_dataset(batch):
            # load and resample audio data from 48 to 16kHz

            audio = batch["audio"]

            # compute log-Mel input features from input audio array
            batch["input_features"] = feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            # compute input length of audio sample in seconds
            batch["input_length"] = len(
                audio["array"]) / audio["sampling_rate"]

            # Get the language of our audio/text
            tokenizer.set_prefix_tokens(
                language=batch["language"], task="transcribe")

            # encode target text to label ids
            batch["labels"] = tokenizer(batch["sentence"]).input_ids

            return batch

        def is_audio_in_length_range(length):
            return length > args.min_input_length and length < args.max_input_length

        def is_labels_in_length_range(labels):
            return len(labels) < max_label_length

        languages = {
            "common_voice_gl": "gl",
            "common_voice_es": "es",
            "common_voice_pt": "pt",
            "common_voice_fr": "fr",
            "common_voice_de": "de",
            "common_voice_en": "en"
        }

        common_voice_gl = DatasetDict()
        common_voice_es = DatasetDict()
        common_voice_pt = DatasetDict()
        common_voice_fr = DatasetDict()
        common_voice_de = DatasetDict()
        common_voice_en = DatasetDict()

        common_voice_gl["train"] = load_dataset(
            "mozilla-foundation/common_voice_13_0", "gl", split="train", use_auth_token=True)
        common_voice_gl["test"] = load_dataset(
            "mozilla-foundation/common_voice_13_0", "gl", split="test", use_auth_token=True)

        common_voice_es["train"] = load_dataset(
            "mozilla-foundation/common_voice_5_0", "es", split="train", use_auth_token=True)
        common_voice_es["test"] = load_dataset(
            "mozilla-foundation/common_voice_5_0", "es", split="test", use_auth_token=True)

        common_voice_pt["train"] = load_dataset(
            "mozilla-foundation/common_voice_7_0", "pt", split="train", use_auth_token=True)
        common_voice_pt["test"] = load_dataset(
            "mozilla-foundation/common_voice_7_0", "pt", split="test", use_auth_token=True)

        common_voice_fr["train"] = load_dataset(
            "mozilla-foundation/common_voice_1_0", "fr", split="train", use_auth_token=True)
        common_voice_fr["test"] = load_dataset(
            "mozilla-foundation/common_voice_1_0", "fr", split="test", use_auth_token=True)

        common_voice_de["train"] = load_dataset(
            "mozilla-foundation/common_voice_2_0", "de", split="train", use_auth_token=True)
        common_voice_de["test"] = load_dataset(
            "mozilla-foundation/common_voice_2_0", "de", split="test", use_auth_token=True)

        common_voice_en["train"] = load_dataset(
            "mozilla-foundation/common_voice_1_0", "en", split="train", use_auth_token=True)
        common_voice_en["test"] = load_dataset(
            "mozilla-foundation/common_voice_1_0", "en", split="test", use_auth_token=True)

        languages_datasets_list = []
        languages_datasets_list = [common_voice_es, common_voice_pt,
                                   common_voice_fr, common_voice_de, common_voice_en]

        if args.prune_well_datasets:
            for language_dataset in languages_datasets_list:

                if len(language_dataset["train"]) < len(common_voice_gl["train"]):
                    continue
                else:
                    random_indices = random.sample(
                        range(len(language_dataset["train"])), len(common_voice_gl["train"]))
                    language_dataset["train"] = language_dataset["train"].select(
                        random_indices)

                    random_indices = random.sample(
                        range(len(common_voice_gl["test"])), len(common_voice_gl["test"]))
                    language_dataset["test"] = language_dataset["test"].select(
                        random_indices)

        columns_to_remove = ["accent", "age", "client_id", "down_votes",
                             "gender", "locale", "path", "segment", "up_votes"]

        # Remove unnecessary columns and add language column
        for dataset_name, language_code in languages.items():
            dataset = globals()[dataset_name]
            for split in dataset:
                language_column = [language_code] * len(dataset[split])
                dataset[split] = dataset[split].add_column(
                    "language", language_column)
            dataset = dataset.remove_columns(columns_to_remove)

        if args.augment_gl_data:
            # augment gl training data
            augmented_raw_gl_training_dataset = common_voice_gl["train"].map(
                augment_dataset, num_proc=16, desc="augment train dataset")

            # combine augmented gl training data
            common_voice_gl["train"] = concatenate_datasets(
                [common_voice_gl["train"], augmented_raw_gl_training_dataset])
            common_voice_gl["train"] = common_voice_gl["train"].shuffle(
                seed=10)

        common_voice["train"] = concatenate_datasets(
            [common_voice_gl["train"], common_voice_es["train"], common_voice_pt["train"], common_voice_fr["train"], common_voice_de["train"], common_voice_en["train"]])
        common_voice["train"] = common_voice["train"].shuffle(seed=10)

        common_voice["test"] = concatenate_datasets(
            [common_voice_gl["test"], common_voice_es["test"], common_voice_pt["test"], common_voice_fr["test"], common_voice_de["test"], common_voice_en["test"]])
        common_voice["test"] = common_voice["test"].shuffle(seed=10)

        common_voice = common_voice.map(remove_unnecessary_spaces)

        common_voice = common_voice.cast_column(
            "audio", Audio(sampling_rate=16000))

        common_voice = common_voice.map(
            prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=16)

        common_voice["train"] = common_voice["train"].filter(
            is_audio_in_length_range, num_proc=8, input_columns=["input_length"]
        )

        common_voice["train"] = common_voice["train"].filter(
            is_labels_in_length_range, num_proc=8, input_columns=["labels"]
        )

        if args.save_dataset_to_disk:
            common_voice.save_to_disk(args.dataset_dir)

    processor.save_pretrained(args.output_dir)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric_asr = evaluate.load("wer")

    def compute_metrics_asr(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * \
            metric_asr.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str("./" + args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        evaluation_strategy=args.evaluation_strategy,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=25,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=4,
        push_to_hub=False,
    )

    class CustomTrainer(Seq2SeqTrainer):

        def __init__(self, alpha_initial, alpha_final, num_min_steps, num_total_steps, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.alpha_initial = alpha_initial
            self.alpha_final = alpha_final
            self.num_min_steps = num_min_steps
            self.num_total_steps = num_total_steps

        def calculate_linear_wieght(self, current_step):
            if current_step <= self.num_min_steps:
                language_weight = self.alpha_initial
            else:
                progress_ratio = (current_step - self.num_min_steps) / \
                    (self.num_total_steps - self.num_min_steps)
                language_weight = self.alpha_initial + \
                    (self.alpha_final - self.alpha_initial) * progress_ratio

            return language_weight

        def compute_loss(self, model, inputs, return_outputs=False):

            labels = inputs.get("labels")
            language_labels = labels[:, 1]

            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")

            current_step = self.state.global_step

            # Index for galician labels
            empty_label_indices = torch.nonzero(
                language_labels == 50319).squeeze()

            loss_fct = CrossEntropyLoss()
            loss_asr = loss_fct(
                logits.view(-1, self.model.config.vocab_size), labels.reshape(-1))

            if args.linear_progressive_weight:

                if current_step <= self.num_min_steps:
                    loss_def = loss_asr
                else:
                    language_weight = self.calculate_linear_wieght(
                        current_step)
                    weights = torch.ones_like(
                        language_labels, dtype=torch.float)
                    weights[empty_label_indices] = language_weight
                    loss_def = (loss_asr*weights).mean()

            elif args.dynamic_weight_adaptation:

                low_losses = []
                well_losses = []

                loss_fct_da = CrossEntropyLoss(reduction='none')
                loss_asr_da = loss_fct_da(
                    logits.view(-1, self.model.config.vocab_size), labels.reshape(-1))

                losses = torch.split(
                    loss_asr_da, args.per_device_train_batch_size)

                weights = torch.ones_like(language_labels, dtype=torch.float)
                weights[empty_label_indices] = args.alpha

                for loss_item, weight_item in zip(losses, weights):
                    if weight_item == args.alpha:
                        low_losses.append(loss_item)
                    else:
                        well_losses.append(loss_item)

                if low_losses:
                    mean_low_losses = torch.stack(low_losses).mean()
                    mean_well_losses = torch.stack(well_losses).mean()

                    if (mean_low_losses * args.alpha) / (mean_well_losses) < 1.0:
                        language_weight = 1.0
                    else:
                        loss_ratio = mean_low_losses / mean_well_losses
                        language_weight = max(args.alpha, loss_ratio.item())

                    loss_def = loss_asr * language_weight
                else:
                    loss_def = loss_asr

            else:
                loss_def = loss_asr

            return (loss_def, outputs) if return_outputs else loss_def

    trainer = CustomTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_asr,
        tokenizer=processor.feature_extractor,
        alpha_initial=args.alpha_initial,
        alpha_final=args.alpha_final,
        num_min_steps=args.num_min_steps,
        num_total_steps=args.num_total_steps,
    )

    trainer.train()


if __name__ == "__main__":
    main()
