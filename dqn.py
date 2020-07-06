import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from PIL import Image
import cv2  # opencv
import sys
import io
import time
import pandas as pd
import numpy as np
from IPython.display import clear_output
from random import randint
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

import random
import pickle
from io import BytesIO
import base64
import json
import time

generation_score = []

game_url = "chrome://dino"
chromebrowser_path = "C:\\Users\\Kantoula\\Desktop\\chromedriver_win32\\chromedriver.exe"

init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"


class Game:
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self.browser = webdriver.Chrome(executable_path=chromebrowser_path, chrome_options=chrome_options)
        self.browser.set_window_position(x=-10, y=0)
        self.browser.get('chrome://dino')
        self.browser.execute_script("Runner.config.ACCELERATION=0")
        self.browser.execute_script(init_script)
        self.browser.implicitly_wait(30)
        self.browser.maximize_window()

    def get_crashed(self):
        return self.browser.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self.browser.execute_script("return Runner.instance_.playing")

    def restart(self):
        self.browser.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def press_right(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_RIGHT)

    def get_score(self):
        score_array = self.browser.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)

    def get_highscore(self):
        score_array = self.browser.execute_script("return Runner.instance_.distanceMeter.highScore")
        for i in range(len(score_array)):
            if score_array[i] == '':
                break
        score_array = score_array[i:]
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        return self.browser.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self.browser.execute_script("return Runner.instance_.play()")

    def end(self):
        self.browser.close()


class DinoAgent:
    def __init__(self, game):
        self.dinoGame = game;
        self.jump();

    def is_running(self):
        return self.dinoGame.get_playing()

    def is_crashed(self):
        return self.dinoGame.get_crashed()

    def jump(self):
        self.dinoGame.press_up()

    def duck(self):
        self.dinoGame.press_down()

    def DoNothing(self):
        self.dinoGame.press_right()


class Game_state:
    def __init__(self, agent, game):
        self._agent = agent
        self.dinoGame = game
        self._display = show_img()
        self._display.__next__()

    def get_next_state(self, actions):
        score = self.dinoGame.get_score()
        high_score = self.dinoGame.get_highscore()

        reward = 0.1
        is_over = False
        if actions[0] == 1:
            self._agent.jump()
        elif actions[1] == 1:
            self._agent.duck()
        elif actions[2] == 1:
            self._agent.DoNothing()

        image = screenshot(self.dinoGame.browser)
        self._display.send(image)

        if self._agent.is_crashed():
            generation_score.append(score)
            time.sleep(0.1)
            self.dinoGame.restart()
            reward = -1
            is_over = True

        image = image_to_tensor(image)
        return image, reward, is_over, score, high_score


def screenshot(browser):
    image_b64 = browser.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)
    return image


def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[:500, :600]
    image = cv2.resize(image, (84, 84))
    image[image > 0] = 255
    image = np.reshape(image, (84, 84, 1))
    return image


def image_to_tensor(image):
    image = np.transpose(image, (2, 0, 1))
    image_tensor = image.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor


def show_img(graphs=False):
    while True:
        screen = (yield)
        window_title = "Dino Agent"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 3
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 5000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def train(model, start):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    game = Game()
    dino = DinoAgent(game)
    game_state = Game_state(dino, game)

    replay_memory = []

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, score, high_score = game_state.get_next_state(action)

    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)  # stacking 4 images

    print("printing size of input state at 0")
    print(state.size())

    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    while iteration < model.number_of_iterations:
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        random_action = random.random() <= epsilon
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        action[action_index] = 1

        image_data_1, reward, terminal, score, high_score = game_state.get_next_state(action)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        replay_memory.append((state, action, reward, state_1, terminal))

        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        output_1_batch = model(state_1_batch)

        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)
        optimizer.zero_grad()

        y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)

        loss.backward()
        optimizer.step()

        state = state_1

        global generation_score
        if len(generation_score) == 0:
            avg_score = 0
        else:
            avg_score = sum(generation_score) / len(generation_score)

        if iteration % 1000 == 0:
            print("iteration:", iteration, "score:", score, "high_score:", high_score, "elapsed time:",
                  time.time() - start, "epsilon:", epsilon, "action:",
                  action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                  np.max(output.cpu().detach().numpy()), "avg_score:", avg_score)

            generation_score = []

        if iteration % 100000 == 0:
            torch.save(model, "pretrained-model/current_model_" + str(iteration) + ".pth")

        iteration += 1


def test(model):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_state(dino, game)

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, s_, h_ = game_state.get_next_state(action)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        action_index = torch.argmax(output)
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action[action_index] = 1

        image_data_1, reward, terminal, s_, h_ = game_state.get_next_state(action)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        state = state_1


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':

        model = torch.load(
            'pretrained-model/current_model_200000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained-model/'):
            os.mkdir('pretrained-model/')

        model = NeuralNetwork()

        if cuda_is_available:
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main('train')
    print('train')