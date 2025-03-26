python train.py --model=img_gauss_stochastic --yaml=img --optim.algo="ESGD" --max_iter=200 --group=neural_image --name=esgd
python train.py --model=img_gauss_stochastic --yaml=img --optim.algo="Adam" --max_iter=200 --group=neural_image --name=adam

python train.py --model=img_siren_stochastic --yaml=img --optim.algo="ESGD" --max_iter=200 --group=neural_image_siren --name=esgd
python train.py --model=img_siren_stochastic --yaml=img --optim.algo="Adam" --max_iter=200 --group=neural_image_siren --name=adam

python train.py --model=img_wavelet_stochastic --yaml=img --optim.algo="ESGD" --max_iter=200 --group=neural_image_wavelet --name=esgd
python train.py --model=img_wavelet_stochastic --yaml=img --optim.algo="Adam" --max_iter=200 --group=neural_image_wavelet --name=adam

python train.py --model=img_relu_stochastic --yaml=img --optim.algo="ESGD" --optim.ESGD.lr=0.1 --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_image_relu --name=esgd
python train.py --model=img_relu_stochastic --yaml=img --optim.algo="Adam" --optim.Adam.lr=1.e-3 --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_image_relu --name=adam



