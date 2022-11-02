# product-images-captioning

Данный репозиторий содержит код моделей, которые можно использовать для генерации описаний товаров магазинов одежды.

## Данные

Для обучения и оценки качества модели используются данные из соревнования 
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

Из датасета были удалены записи с дублирующимися описаниями и записи, не имеющие соответствующих картинок.

## Модель

Для решения задачи используется архитектура Encoder-Decoder.
В качестве encoder'а используется [vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224),
в качестве decoder'а [distilgpt2](https://huggingface.co/distilgpt2).
