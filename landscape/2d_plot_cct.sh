
# ------------------------------------------------------------
# SGD
#------------------------------------------------------------
mpirun -n 24 python plot_surface.py --x=-8:12:21 --y=-8:12:21 --model cct_2_3x2_32 \
--model_file cifar10/trained_nets/gm/cct.7t \
--dir_file cifar10/trained_nets/gm/PCA_weights/directions.h5 \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter #\
#  --main_dir_model cifar10/trained_nets/gm/1Falsecct.7t


