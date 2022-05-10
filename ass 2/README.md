## Code Originization
- `./utils/ResNet.py`：Define class `CIFAR_ResNet`.
- `./utils/cutout.py`：Realize data aug "cutout".
- `./utils/mixup.py`：Realize data aug "mixup".
- `./utils/cutmix.py`：Realize data aug "cutmix".
- `./main.ipynb`：The process of training the net with different data aug methods defined before, which is run on [kaggle](kaggle.com).
- `./show.ipynb`：Show three example images after different data aug methods defined before.
 - `./load_model.ipynb`：Show how to use the pre-trained & saved model to predicted new data.

 ## How to train, show, and load
 To training from scratch, you may see how the code is orginized in `./main.ipynb`(Note That the model is saved in my [google drive](https://drive.google.com/drive/folders/148frXHSm31UpbejRjVpwg3P67nHjIgaJ), you should download it and put it in the same dir as `./load_model.ipynb`). Or for simply, the key option might be:
```python
pureNet = CIFAR_ResNet(method = "pure",
                    lr = 1e-2, 
                    lr_decay = 0.95,
                    loss_func = nn.CrossEntropyLoss(), 
                    weight_decay = 1e-4, 
                    epochs = 25, 
                    denote_path = "./train_history/pure",
                    verbose = True)
pureNet.train()
pureNet.save("pureNet")
```

To load the pre-trained ResNet-18 model and use it, you may see how the code is orginized in `./load_model.ipynb`. Or for simply, the key option might be:
```python
NetModel = CIFAR_ResNet.load("./models/cutoutNet.pkl")
outputs = NetModel(X_try)
```

Further, if you want to show more example images after different data aug methods, you can run `./show.ipynb` more times.

## Training History
If you are interested in the training history, the dir "train_history" denote the the training history by `TensorBoard`, you can enter:
```
tensorboard --logdir "./train_history/pure"
...
...
tensorboard --logdir "./train_history/cutmix"
```
in the terminal to look them. 



