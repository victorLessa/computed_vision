# Reconhecimento facial

Este repositorio possui treinamento do modelo YOLOv8 para reconhecimento de rost `detection_faces` e reconhecimento de faces `recognizer_faces`

O reconhecimento de faces captura imagem do seu rosto utilizando webcam `preprocess.py` e depois realiza o treinamento do modelo `train.py` para o modelo ter
a capacidade de te identificar

O `app.py` é o projeto final, nele o reconhecimento de realizado e caso ele não conheça o sistema atribuíra um valor de "Intruso" sobre o rosto do individuo.
