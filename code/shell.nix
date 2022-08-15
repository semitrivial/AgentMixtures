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
    (unstable.python39.withPackages (ps: with ps; [
    #(python37.withPackages (ps: with ps; [
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
        pyopengl
        glfw
        pip
        absl-py
    ]))
  ];
  PYTHONPATH="";
  #PYTHONPATH="./python-root";
  #PIP_TARGET="./python-root";
  #PYTHONUSERBASE="./python-root";
  #PYTHONUSERSITE="./python-root";
  LD_LIBRARY_PATH="";

  #virtual env does not work either - pulls numpy and so on
  #shellHook="
  #  python3.9 -m venv python-venv
  #  source python-venv/bin/activate
  #";


  #nix-shell --run 'pip install mujoco --no-cache-dir -t python-root'
  # fails because it check dependencies in PIP_TARGET
}
