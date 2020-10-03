if __name__ == "__main__":
    import gfootball  # noqa

    # train_agent()
    from kaggle_environments import make

    env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle"})

    # This is the observation that is passed on agent function.
    obs = env.state[0]['observation']

    print(obs.keys())
