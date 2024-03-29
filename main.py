import gymnasium as gym
import numpy as np
from actorCritic import ActorCritic
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 1500 scores')
    plt.savefig(figure_file)

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    index = input("index :") or "0"
    alpha = input("alpha (1e-4):") or 1e-4
    gamma = input("gamma (0.99):") or 0.99
    agent = ActorCritic(alpha=float(alpha), gamma=float(gamma), action_space=env.action_space.n, a1=(input("a1 (86):")) or 86, a2=(input("a2 (32):")) or 32, c1=(input("c1 (86):")) or 86, c2=(input("c2 (32):")) or 32, l=(input("count limit:")) or 1, Async=(input("Async :")) or "A", index=index )
    n_games = 1500
    time = datetime.now()

    filePath = "tmp/img"+index+"/LunarLander"+"-"+str(time)+"-g"+str(agent.gamma)+"-a"+str(agent.alpha)+"-a1_"+str(agent.aD1_dims)+"-a2_"+str(agent.aD2_dims)+"-c1_"+str(agent.cD1_dims)+"-c2_"+str(agent.cD2_dims)+"-Async_"+agent.Async+"-LM_"+str(agent.countLimit)
    highScore = env.reward_range[0]
    scoreHistory = []
    loadCheckpoint = True
    print(filePath+".png")

    if loadCheckpoint:
        logDate = input("log date:") or None
        agent.load(logDate)
        n_games = 501
    for i in range (n_games):
        print(i)
        observation = env.reset()[0]
        done = False
        score = 0
        try:
            while not done:
                action, actionProba, probs = agent.chooseAction(observation)
                observation1, reward, done, _, info = env.step(action)
                score += reward
                if not loadCheckpoint:
                    agent.learn(observation, reward, observation1, done)
                elif score <= -500
                    done = True
                observation = observation1
        except (KeyboardInterrupt, AssertionError):
            print(action)
            print(actionProba.sample())
            print(probs)
           
            now = str(datetime.now()-time)
            figureFile = filePath + "-" + now + ".png"
            x = [j+1 for j in range (i)]
            plot_learning_curve(x, scoreHistory, figureFile)

            if loadCheckpoint:
                figureFile1 = filePath + "-" + now + "training" + ".png"
                dataset = pd.DataFrame({'Reward': scoreHistory})
                dataset['Succeded'] = dataset.Reward.apply(lambda x: True if x > 200 else False)
                plot = sns.displot(x="Reward", data=dataset)
                plot.savefig(figureFile1)

            sys.exit()
        
        scoreHistory.append(score)
        averageScore = np.mean(scoreHistory[-100:])

        if averageScore > highScore:
            highScore = averageScore
            if not loadCheckpoint:
                agent.save()
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % averageScore)

    now = str(datetime.now()-time)
    figureFile = filePath + "-" + now + ".png"
    x = [i+1 for i in range (n_games)]
    plot_learning_curve(x, scoreHistory, figureFile)


    if loadCheckpoint:
        figureFile1 = filePath + "-" + now + "training" + ".png"
        dataset = pd.DataFrame({'Reward': scoreHistory})
        dataset['Succeded'] = dataset.Reward.apply(lambda x: True if x > 200 else False)
        plot = sns.displot(x="Reward", data=dataset)
        plot.savefig(figureFile1)



