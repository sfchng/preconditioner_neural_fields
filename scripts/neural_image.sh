## Gaussian ##
python train.py --model=img_gauss_stochastic --yaml=img --optim.algo="ESGD" --max_iter=200 --group=neural_image --name=esgd
python train.py --model=img_gauss_stochastic --yaml=img --optim.algo="Adam" --max_iter=200 --group=neural_image --name=adam
python train.py --model=img_gauss_stochastic --yaml=img --optim.algo="Adahessian" --max_iter=200 --group=neural_image --name=adahessian
python train.py --model=img_gauss_stochastic --yaml=img --optim.algo="Adahessian_J" --max_iter=200 --group=neural_image --name=adahessian_j
python train.py --model=img_gauss_stochastic --yaml=img --optim.algo="Preconditioner_KFAC" --max_iter=200 --group=neural_image --name=precond_kfac

## sine ##
python train.py --model=img_siren_stochastic --yaml=img --optim.algo="ESGD" --max_iter=200 --group=neural_image_siren --name=esgd
python train.py --model=img_siren_stochastic --yaml=img --optim.algo="Adam" --max_iter=200 --group=neural_image_siren --name=adam
python train.py --model=img_siren_stochastic --yaml=img --optim.algo="Adahessian" --max_iter=200 --group=neural_image_siren --name=adahessian
python train.py --model=img_siren_stochastic --yaml=img --optim.algo="Adahessian_J" --max_iter=200 --group=neural_image_siren --name=adahessian_j
python train.py --model=img_siren_stochastic --yaml=img --optim.algo="Preconditioner_KFAC" --max_iter=200 --group=neural_image_siren --name=precond_kfac


## wavelet ##
python train.py --model=img_wavelet_stochastic --yaml=img --optim.algo="ESGD" --max_iter=200 --group=neural_image_wavelet --name=esgd
python train.py --model=img_wavelet_stochastic --yaml=img --optim.algo="Adam" --max_iter=200 --group=neural_image_wavelet --name=adam
python train.py --model=img_wavelet_stochastic --yaml=img --optim.algo="Adahessian" --max_iter=200 --group=neural_image_wavelet --name=adahessian
python train.py --model=img_wavelet_stochastic --yaml=img --optim.algo="Adahessian_J" --max_iter=200 --group=neural_image_wavelet --name=adahessian_j
python train.py --model=img_wavelet_stochastic --yaml=img --optim.algo="Preconditioner_KFAC" --max_iter=200 --group=neural_image_wavelet --name=precond_kfac


## ReLU (PE) ##
python train.py --model=img_relu_stochastic --yaml=img --optim.algo="ESGD" --optim.ESGD.lr=0.1 --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_image_relu --name=esgd
python train.py --model=img_relu_stochastic --yaml=img --optim.algo="Adam" --optim.Adam.lr=1.e-3 --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_image_relu --name=adam
python train.py --model=img_relu_stochastic --yaml=img --optim.algo="Adahessian" --optim.Adahessian.lr=0.1 --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_image_relu --name=adahessian
python train.py --model=img_relu_stochastic --yaml=img --optim.algo="Adahessian_J" --optim.Adahessian.lr=0.1 --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_image_relu --name=adahessian_j
python train.py --model=img_relu_stochastic --yaml=img --optim.algo="Preconditioner_KFAC" --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_image_relu --name=precond_kfac2 --optim.Adam.lr=1.e-3 


