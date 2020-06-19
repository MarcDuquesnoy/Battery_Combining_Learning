Data Driven
========================================================

This folder contains all necessary
informations and ressources for the benchmarking of the Machine Learning
approach within Fortran and Python.

Please, do not hesitate to discover the team of researchers behind the code and also the ARTISTIC project
 involved : [ARTISTIC](https://www.u-picardie.fr/erc-artistic/?L=0)


Setup
========================================================
We ask you to get a virtual environment such as **conda** for the Python exploration.
Go to their webpage and download it. All dependencies will be
installed on your environment.
Because the SISSO method was implemented under Fortran, you need to switch 
between the latter and Python for the analysis.

Benchmarking
=========================================================
1. Launch _preprocessing.py_ with the corresponding path for the associated electrode generation
 dataset.
2. Use the training set corresponding for one output feature for the SISSO
training.
3. Launch _testing-model.py_ with the resulting file and the testing &
 validation set.

Example for one output feature
========================================================

Inside this folder, SISSO ressources are provided for only **one** output feature
(i.e. Active surface), in order to avoid repetitions of files.

The example given provides :
1. SISSO.out
2. SISSO.in
2. ./dat/train.dat
3. ./dat/test.dat
4. sisso.x

SISSO
========================================================

For the training of the SISSO, we strongly encourage you to follow the 
procedure made by [Rouyang et al.](https://github.com/rouyang2017/SISSO) from Github
repository where you will find all dependencies for a correct use of the method under Fortran.
Please, note that the corresponding **train.dat** represents one of the 
six training files coming from the preprocessing step above.

In a shell, launch the command line below to get the output file :
```shell script
mpirun -np xxx sisso.x >log
```
 
where 'xxx' is the number of processors used, and 'sisso.x' the corresponding
executable.

_SISSO.out_ is the output file that contains informations of the training with
descriptors, coefficients and Root Mean Square Error (RMSE) for each dimensional projection. The one that
minimize the RMSE will be chosen.

**Note** : if the generation of the SISSO executable gets issues, please use the one
locate inside this folder. 

Testing
========================================================
_testing-model.py_ has to test the SISSO analysis.
After changing the path of the testing set associated to the output
feature, it will calculate the goodness of fitting through the R2 score.
Also, it uses a validation dataset with data never seen in the initial dataset.

that is why, you can launch the following command line in the corresponding
 repository:
```shell script
python testing-model.py
```


 Authors
 ========================================================
  - **Marc DUQUESNOY** , Research Engineer
  
 Contributing 
========================================================

Pull requests are not allowed. For more informations about the code, please
 contact the authors.
 Do not hesitate to ask them if inappropriate bugs occur.
 
 
 License
========================================================

This project is licensed under the [GNU Public License](https://www.gnu.org/licenses/gpl-3.0.en.html)

Contributors
========================================================
- **Marc DUQUESNOY** , Research Engineer
- **Teo Lombardo** , PhD 
- **Mehdi Chouchane** , PhD
- **Emiliano Primo** , Post-Doc