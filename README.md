# Kaggle football

# Setup

## GFootball dependencies
### Use precompiled GFootball version
As per setup shown in https://www.kaggle.com/piotrstanczyk/gfootball-template-bot

```bash
sudo ./setup/install_precompiled.sh
```

### Alternative setup
Compile manually - try if above setup fails

```bash
sudo ./setup/install_dependencies.sh
```

## Python env

```bash
pip install -r requirements.txt
```

# Building agents
For custom agents, the path should be to a .py file that defines the function agent. This should take the environment observations as an input and return a single action (int) in a list. 
```python
from typing import List

def agent(obs) -> List[int]:
    action = 1
    return [action]   
```

See example_agent.py for an example using the template agent show here: https://www.kaggle.com/c/google-football/overview/getting-started
See random_agent.py for an example that imports the agent from a module defined here.  

# Running
Agents can be run against each other using the env.run method. This expects 2 agents as input, which can be a custom path or predefined agent.

See scripts/run.py for an example. Not that if using a relative path, the working directory of the script should be set to the top level of the project. If using an IDE, this can be set in the run profile. Or it can be run from the command line:

```bash
python3 -m scripts.run_agent_vs_agent
```

Output should look something like:
```bash
>>> ...
>>> Step 250
    Left player (random_agent.py): 
    actions taken: [16], reward: 0, status: ACTIVE, info: {}
    Left player (run_right): 
    actions taken: [5], reward: 0, status: ACTIVE, info: {}

    Step 251
    Right player (random_agent.py): 
    actions taken: [9], reward: 0, status: ACTIVE, info: {}
    Right player (run_right): 
    actions taken: [5], reward: 0, status: ACTIVE, info: {}
>>> ...
```

# Creating submission
Submissions can be made as single .py files or as a .tar.gz containing multiple files.

## As single files
If everything is self contained in one .py file, it can either be upload manually or via the Kaggle API to submit (this is installed in the requirements here, but also requires setting up an API token if not used before - see https://github.com/Kaggle/kaggle-api)

```bash
kaggle competitions submit -c google-football -f example_agent.py -m "Test submit of example agent"
```

## Model weights in single file
It's possible to serialise model weights to a string and define them as a contestant inside the [agent].py. It's cumbersome, but it avoids problems trying to reload model weights to disk and uploaded in a .tar.gz (see "Multiple files a zip below)" )

1) Train example model in an interactive console. This creates the variable weights_str, which is a compressed, serialized, version of the model weights, which is created with the code:
   ```
   weights_str = zlib.compress(pickle.dumps([keras model].get_weights()))
   ```

2) Manually copy and paste the contents of weights_str into main.py, in constant WEIGHTS_STR. eg:
   ```python
   WEIGHTS_STR = 'Serialised gibberish goes here'
    
    # These weights are deserialized and set with:
   [keras model].set_weights(pickle.loads(zlib.decompress(WEIGHTS_STR)))
   ```
   
3) And submit
    ```bash
    kaggle competitions submit -c google-football -f main.py -m "Test submit of main.py containing model weights and model code"
    ```

## Multiple files as zip (not working?)

For example, agent using neural network (here saved into folder test_model). 

1) Train example model
    ```bash
    python3 -m scripts.train_model
    ```
   This will output to test_model/
2) Create main.py that defines the agent - see main.py in here as example. This loads model saved in test_model/.
3) Package the required files
    ```bash
    tar -czvf submit.tar.gz main.py test_model
    ```
4) And submit
    ```bash
    kaggle competitions submit -c google-football -f submit.tar.gz -m "Test submit of example RL agent"
    ```
