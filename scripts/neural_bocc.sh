

## sine ##
python train.py --model=occupancy_siren_stochastic --yaml=bocc --optim.algo="ESGD" --max_iter=200 --group=neural_bocc_siren --name=esgd
python train.py --model=occupancy_siren_stochastic --yaml=bocc --optim.algo="Adam" --max_iter=200 --group=neural_bocc_siren --name=adam
python train.py --model=occupancy_siren_stochastic --yaml=bocc --optim.algo="Adahessian" --max_iter=200 --group=neural_bocc_siren --name=adahessian
python train.py --model=occupancy_siren_stochastic --yaml=bocc --optim.algo="Adahessian_J" --max_iter=200 --group=neural_bocc_siren --name=adahessian_j
python train.py --model=occupancy_siren_stochastic --yaml=bocc --optim.algo="Preconditioner_KFAC" --max_iter=200 --group=neural_bocc_siren --name=precond_kfac

## wavelet ##
python train.py --model=occupancy_wavelet_stochastic --yaml=bocc --optim.algo="ESGD" --max_iter=200 --group=neural_bocc_wavelet --name=esgd
python train.py --model=occupancy_wavelet_stochastic --yaml=bocc --optim.algo="Adam" --max_iter=200 --group=neural_bocc_wavelet --name=adam
python train.py --model=occupancy_wavelet_stochastic --yaml=bocc --optim.algo="Adahessian" --max_iter=200 --group=neural_bocc_wavelet --name=adahessian
python train.py --model=occupancy_wavelet_stochastic --yaml=bocc --optim.algo="Adahessian_J" --max_iter=200 --group=neural_bocc_wavelet --name=adahessian_j
python train.py --model=occupancy_wavelet_stochastic --yaml=bocc --optim.algo="Preconditioner_KFAC" --max_iter=200 --group=neural_bocc_wavelet --name=precond_kfac

## ReLU (PE) ##
python train.py --model=occupancy_relu_stochastic --yaml=bocc --optim.algo="ESGD" --optim.ESGD.lr=0.1 --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_bocc_relu --name=esgd
python train.py --model=occupancy_relu_stochastic --yaml=bocc --optim.algo="Adam" --optim.Adam.lr=1.e-3 --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_bocc_relu --name=adam
python train.py --model=occupancy_relu_stochastic --yaml=bocc --optim.algo="Adahessian" --optim.Adahessian.lr=0.1 --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_bocc_relu --name=adahessian
python train.py --model=occupancy_relu_stochastic --yaml=bocc --optim.algo="Adahessian_J" --optim.Adahessian.lr=0.1 --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_bocc_relu --name=adahessian_j
python train.py --model=occupancy_relu_stochastic --yaml=bocc --optim.algo="Preconditioner_KFAC" --arch.relu.posenc.enabled=True --max_iter=200 --group=neural_bocc_relu --name=precond_kfac2 --optim.Adam.lr=1.e-3 

## Gaussian ##
python train.py --model=occupancy_gauss_stochastic --yaml=bocc --optim.algo="ESGD" --max_iter=200 --group=neural_bocc --name=esgd
python train.py --model=occupancy_gauss_stochastic --yaml=bocc --optim.algo="Adam" --max_iter=200 --group=neural_bocc --name=adam
python train.py --model=occupancy_gauss_stochastic --yaml=bocc --optim.algo="Adahessian" --max_iter=200 --group=neural_bocc --name=adahessian
python train.py --model=occupancy_gauss_stochastic --yaml=bocc --optim.algo="Adahessian_J" --max_iter=200 --group=neural_bocc --name=adahessian_j
python train.py --model=occupancy_gauss_stochastic --yaml=bocc --optim.algo="Preconditioner_KFAC" --max_iter=200 --group=neural_bocc --name=precond_kfac