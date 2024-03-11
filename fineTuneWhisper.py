from datasets import load_dataset, DatasetDict
import pandas
from functools import partial
def main():
    audio_set = load_dataset("json", data_files = "test.json")

    train_testvalid = audio_set["train"].train_test_split(test_size=0.2)

    #test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

    audio_set = DatasetDict({
        'train': train_testvalid['train'],
        'test': train_testvalid['test']})

    audio_set = audio_set.remove_columns(["Name"])

    print(audio_set)

    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")


    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="English", task="transcribe")

    import librosa
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["Path"]
        print(batch["Path"])

        signal, sr = librosa.load(audio, sr=16000)

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(signal, sampling_rate=sr).input_features[0]

        # encode target text to label ids 
        normalizer = BasicTextNormalizer()
        batch["labels"] = tokenizer(normalizer(batch["Transcript"])).input_ids
        return batch

    audio_set = audio_set.map(prepare_dataset, remove_columns=audio_set.column_names["train"], num_proc=4)

    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="English", task="transcribe")

    import torch



    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    class DataCollatorSpeechSeq2SeqWithPadding:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    import evaluate
    from transformers import WhisperForConditionalGeneration
    from transformers import Seq2SeqTrainingArguments
    from transformers import Seq2SeqTrainer
    

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        with open('refs_and_preds.txt', 'w') as f:
            for ref, pred in zip(label_str, pred_str):
                f.write(f"Ref: {ref}\n")
                f.write(f"Pred: {pred}\n\n")

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

    model.generation_config.language = "en"
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-large-v3-finetuned-1",  # change to a repo name of your choice
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        num_train_epochs = 10,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        per_device_eval_batch_size = 8,
        generation_max_length = 225,
        logging_steps = 1,
        load_best_model_at_end=True,
        push_to_hub=False,
        save_steps = 0.1,
        predict_with_generate = True,
        metric_for_best_model="wer",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=audio_set["train"],
        eval_dataset=audio_set["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    model.config.use_cache = False
    processor.save_pretrained(training_args.output_dir)

    trainer.train()

if __name__ == '__main__':
    main()