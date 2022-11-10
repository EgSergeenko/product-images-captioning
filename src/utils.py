import torch


def get_max_length(texts, tokenizer):
    max_length = 0
    for text in texts:
        text = add_special_tokens(
            text,
            tokenizer.bos_token,
            tokenizer.eos_token,
        )
        labels = tokenizer(
            text,
            return_tensors='pt',
        ).input_ids
        max_length = max(max_length, labels.size()[-1])
    return max_length


def get_predictions(
    model,
    tokenizer,
    data_loader,
    device,
):
    predictions = []
    model.eval()
    for idx, batch in enumerate(data_loader):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                max_length=labels.size()[-1],
                return_dict_in_generate=True,
            ).sequences

        prediction = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        predictions.extend(prediction)

    return predictions


def add_special_tokens(
    text: str,
    bos_token: str,
    eos_token: str,
) -> str:
    return '{0} {1} {2}'.format(
        bos_token,
        text,
        eos_token,
    )
