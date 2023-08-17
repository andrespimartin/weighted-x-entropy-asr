# Weighted Cross-entropy for Low-Resource Languages in Multilingual Speech Recognition

This repository contains code for the paper titled "Weighted Cross-entropy for Low-Resource Languages in Multilingual Speech Recognition". The paper addresses the challenge of integrating low-resource languages into multilingual automatic speech recognition (ASR) systems.

## Abstract 📑

This paper presents a novel approach to integrating low-resource languages into pre-trained multilingual automatic speech recognition (ASR) models. The key innovation lies in the application of weighted cross-entropy, a technique typically used for unbalanced datasets, to facilitate the incorporation of low-resource languages. The proposed method is integrated into the Whisper multilingual ASR model, which is fine-tuned on both well-resource and low-resource languages. Language-weighted dynamic cross-entropy and data augmentation are employed in the training process. The results demonstrate a remarkable 6.69% reduction in word error rate (WER) for the low-resource language compared to the fine-tuned model without applying the proposed approach. Furthermore, the WER is reduced by 48.86% compared to the original Whisper model. Notably, the approach yields an average WER reduction of 3.29% across all six languages, indicating no degradation for the well-resourced languages.

## Code Overview 📁

The repository includes code for the data preprocessing, augmentation, training, and evaluation of the Whisper multilingual ASR model. It also provides scripts for fine-tuning the model with weighted cross-entropy and language-specific data augmentation. Additionally, the repository contains dataset manipulation scripts, model configuration files, and example usage scripts.

## Usage 🚀

1. **Clone this repository:**

    ```bash
    git clone https://github.com/andrespimartin/wce-low-resource-language-multilingual-asr.git
    cd wce-low-resource-language-multilingual-asr
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Just run the `.sh` file:**

    ```bash
    ./run.sh
    ```

## Setup ⚙️

The code is prepared for multilingual training on the languages of the paper. Feel free to modify or add new languages.

### Basic Configuration

The basic configuration of the training and data set can be set in the `run.sh` file:


```bash
--base_model_dir "openai/whisper-small" \
--output_dir "your_location" \
--dataset_dir "your_dataset_location" \
--load_dataset_from_disk True \
--save_dataset_to_disk False \
--prune_well_datasets False \
--augment_gl_data False \
--max_input_length 30 \
--min_input_length 0 \
```

### Training Parameters

Below are the training-related parameters that can be configured:

```bash
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-5 \
--weight_decay 0.01 \
--warmup_steps 800 \
--max_steps 8000 \
--gradient_checkpointing True \
--fp16 True \
--evaluation_strategy "steps" \
--per_device_eval_batch_size 8 \
--predict_with_generate True \
--generation_max_length 225 \
--save_steps 1000 \
--eval_steps 1000 \
```

### Weighted Cross-Entropy

There are two Weighted Cross-Entropy configuration options. If `linear_progressive_weight` is `True`, you must set the parameters `alpha_initial`, `alpha_final`, `num_min_steps` and `num_total_steps`. If `dynamic_weight_adaptation` is `True`, you must set the value of `alpha`. For more information, see the paper.

```bash
--alpha_initial 2.0 \
--alpha_final 5.0 \
--num_min_steps 4000 \
--num_total_steps 8000 \
--linear_progressive_weight True \
--alpha 1.5 \
--dynamic_weight_adaptation False
```

## Citation 📖

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{pineiro2023weightedasr,
  title={Weighted Cross-entropy for Low-Resource Languages in Multilingual Speech Recognition},
  author={Piñeiro-Martín, Andrés, García-Mateo, Carmen, Docío-Fernández, Laura and López-Pérez, María del Carmen},
  journal={xxxxxx},
  year={2023}
}
```

## License 🔒

This project is licensed under the [Apache License 2.0](LICENSE)
