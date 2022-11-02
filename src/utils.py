import torch


def get_max_length(texts, tokenizer):
    max_length = 0
    for text in texts:
        labels = tokenizer(
            text,
            return_tensors='pt',
        ).input_ids.squeeze()
        max_length = max(max_length, labels.size()[0])
    return max_length


def get_prediction(
    model,
    tokenizer,
    pixel_values,
    max_length=64,
    num_beams=4,
):
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values.unsqueeze(0),
            max_length=max_length,
            num_beams=num_beams,
            return_dict_in_generate=True,
        ).sequences.detach().cpu()
    prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [token.strip() for token in prediction]