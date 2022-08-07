with import <nixpkgs> {};
let
  unstable = import <nixos-unstable> {};
in mkShell {
  #nativeBuildInputs = [
  #  unstable.cudatoolkit_11_6
  #];
  buildInputs = [
    #lm_sensors
    #moreutils
    #cudatoolkit
    #unstable.python39Packages.pytorchWithCuda
    #unstable.cudatoolkit_11_6
    #(unstable.python39.withPackages (ps: with ps; [
    (python37.withPackages (ps: with ps; [
        gym
        #cython
        numpy
        #opencv4 #for stable_baselines
        #tensorflowWithoutCuda #tensorflow-probability #patch stable_baselines to remove tensorflow.contrib
        #baselines
        pytorchWithoutCuda
        # pytorchWithCuda
        # For saving models
        cloudpickle
        # For reading logs
        pandas
        # Plotting learning curves
        matplotlib

        pyglet

        #huggingface-hub
        scipy
        #optuna
        colorlog
        sqlalchemy
        #cmaes
        tqdm
        filelock
        wasabi
        seaborn
        pytest
        #utils
        #pybullet
        #pygame_sdl2
        pip
    ]))
  ];
  PYTHONPATH="./python-root";
  LD_LIBRARY_PATH="";
}
