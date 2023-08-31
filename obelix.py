import cv2
import random
import argparse

import numpy as np

class OBELIX:
    def __init__(self, scaling_factor):
        self.frame_size = (500, 500, 3)
        self.frame = np.ones(self.frame_size, np.uint8) * 0
        self.bot_radius = int(scaling_factor * 12 / 2)            # 12" diameter
        self.facing_angle = 0

        self.bot_center_x = 200
        self.bot_center_y = 200
        self.bot_color = (255, 255, 255)

        self.move_options = {"L45": 45, "L22": 22.5, "FW": 0, "R22": -22.5, "R45": -45}
        self.forward_step_unit = 5

        self.sonar_fov = 20
        self.sonar_far_range = 30 * scaling_factor
        self.sonar_near_range = 18 * scaling_factor
        self.sonar_range_offset = 9 * scaling_factor
        self.sonar_positions = [-90-22, -90+22, -45, -22, 22, 45, 90-22, 90+22]
        self.sonar_facing_angles = [-90, -90, 0, 0, 0, 0, 90, 90]

        self.ir_sensor_range = 4 * scaling_factor

        self.reward = 0
        self.sensor_feedback = np.zeros(18)
        self.sensor_feedback_masks = np.zeros((9, self.frame_size[0], self.frame_size[1]), np.uint8)
        self.stuck_flag = 0

        self.box_size = int(12 * scaling_factor)
        self.box_center_x = np.random.randint(self.box_size, self.frame_size[1] - self.box_size, 1, int)[0]
        self.box_center_y = np.random.randint(self.box_size, self.frame_size[0] - self.box_size, 1, int)[0]
        self.box_yaw_angle = 30#np.random.randint(0, 90, 1, int)[0]
        self.box_corners = []
        self.box_frame = np.zeros(self.frame_size, np.uint8)
        # cv2.circle(self.box_frame, (self.box_center_x, self.box_center_y), 2, self.bot_color, 1)
        for i in range(0, 360, 90):
            x = self.box_center_x + (self.box_size // 2) * np.cos(np.deg2rad(self.box_yaw_angle + i))
            y = self.box_center_y + (self.box_size // 2) * np.sin(np.deg2rad(self.box_yaw_angle + i))
            self.box_corners.append([x, y])
        cv2.fillPoly(self.box_frame, np.array([self.box_corners],
                                              dtype=np.int32), (100, 100, 100))
        self.box_frame = cv2.flip(self.box_frame, 0)
        cv2.rectangle(self.box_frame, (0 + 10, 0 + 10), (self.frame_size[0] - 10, self.frame_size[1] - 10), (100, 100, 100), 1)
        #   *********************Self-Test***********************************
        self.neg_circle_frame = np.zeros(self.frame_size, np.uint8)
        self.neg_circle_center_x = np.random.randint(self.box_size, self.frame_size[1] - self.box_size, 1, int)[0]
        self.neg_circle_center_y = np.random.randint(self.box_size, self.frame_size[0] - self.box_size, 1, int)[0]
        # cv2.circle(self.neg_circle_frame, (self.neg_circle_center_x, self.neg_circle_center_y), 1*scaling_factor, (100, 100, 100), -1)
        #   *****************************************************************
        self.bot_mask = np.ones(self.frame_size, np.uint8) * 0
        self.done = False
        self.enable_push = False
        self.active_state = 'F'

    def render_frame(self):
        self.frame = np.ones(self.frame_size, np.uint8) * 0
        self.bot_mask = np.ones(self.frame_size, np.uint8) * 0
        cv2.rectangle(self.frame, (0+5, 0+5), (self.frame_size[0]-5, self.frame_size[1]-5), (255, 0, 0), 1)
        cv2.rectangle(self.frame, (0+10, 0+10), (self.frame_size[0]-10, self.frame_size[1]-10), (255, 0, 0), 1)

        self.box_frame = np.zeros(self.frame_size, np.uint8)
        self.box_corners = []
        # cv2.circle(self.box_frame, (self.box_center_x, self.box_center_y), 2, self.bot_color, 1)
        for i in range(0, 360, 90):
            x = self.box_center_x + (self.box_size // 2) * np.cos(np.deg2rad(self.box_yaw_angle + i))
            y = self.box_center_y + (self.box_size // 2) * np.sin(np.deg2rad(self.box_yaw_angle + i))
            self.box_corners.append([x, y])
        cv2.fillPoly(self.box_frame, np.array([self.box_corners],
                                              dtype=np.int32), (100, 100, 100))

        self.sensor_feedback_masks = np.zeros((9, self.frame_size[0], self.frame_size[1]), np.uint8)

        cv2.circle(self.frame, (self.bot_center_x, self.bot_center_y), self.bot_radius, self.bot_color, 1)
        cv2.circle(self.bot_mask, (self.bot_center_x, self.bot_center_y), self.bot_radius, (100, 100, 100), -1)
        # self.bot_mask = cv2.flip(self.bot_mask, 0)
        # cv2.imshow("bot_mask", self.bot_mask)
        # cv2.imshow("box_frame", self.box_frame)

        for sonar_range, sonar_intensity in zip([self.sonar_far_range, self.sonar_near_range, self.sonar_range_offset], [100, 50, 0]):
            for index, (sonar_pos_angle, sonar_face_angle) in enumerate(zip(self.sonar_positions, self.sonar_facing_angles)):
                if sonar_intensity == 0:
                    noise_reduction = 2
                else:
                    noise_reduction = 0
                p1_x = self.bot_center_x + self.bot_radius * np.cos(np.deg2rad(self.facing_angle + sonar_pos_angle))
                p1_y = self.bot_center_y + self.bot_radius * np.sin(np.deg2rad(self.facing_angle + sonar_pos_angle))
                p2_x = p1_x + sonar_range * np.cos(
                    np.deg2rad(self.facing_angle + sonar_face_angle + self.sonar_fov // 2 + noise_reduction))
                p2_y = p1_y + sonar_range * np.sin(
                    np.deg2rad(self.facing_angle + sonar_face_angle + self.sonar_fov // 2 + noise_reduction))
                p3_x = p1_x + sonar_range * np.cos(
                    np.deg2rad(self.facing_angle + sonar_face_angle - self.sonar_fov // 2 - noise_reduction))
                p3_y = p1_y + sonar_range * np.sin(
                    np.deg2rad(self.facing_angle + sonar_face_angle - self.sonar_fov // 2 - noise_reduction))

                cv2.fillPoly(self.frame, np.array([[[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]]], dtype=np.int32), sonar_intensity)
                cv2.fillPoly(self.sensor_feedback_masks[index],
                             np.array([[[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]]], dtype=np.int32), sonar_intensity)

        p1_x = int(self.bot_center_x + self.bot_radius * np.cos(np.deg2rad(self.facing_angle)))
        p1_y = int(self.bot_center_y + self.bot_radius * np.sin(np.deg2rad(self.facing_angle)))
        p2_x = int(p1_x + self.ir_sensor_range * np.cos(np.deg2rad(self.facing_angle)))
        p2_y = int(p1_y + self.ir_sensor_range * np.sin(np.deg2rad(self.facing_angle)))
        cv2.line(self.frame, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)
        cv2.line(self.sensor_feedback_masks[8], (p1_x, p1_y), (p2_x, p2_y), (50, 50, 50), 2)

        self.frame = cv2.addWeighted(self.frame, 1.0, self.box_frame, 1.0, 0)
        self.frame = cv2.addWeighted(self.frame, 1.0, self.neg_circle_frame, 1.0, 0)
        self.frame = cv2.flip(self.frame, 0)

        # feedback_image = np.asarray(self.sensor_feedback * 255, np.uint8).reshape(self.sensor_feedback.shape[0], 1).T
        # feedback_image = cv2.resize(feedback_image, (frame.shape[1], 20), cv2.INTER_NEAREST)
        # cv2.imshow("feedback_sensor_image", feedback_image)
        cv2.imshow("Experiment Environment (Behaviour 1: Finding a Box)", self.frame)
        cv2.waitKey(1)
        # for i in range(9):
        #     cv2.imshow("mask" + str(i), cv2.flip(self.sensor_feedback_masks[i], 0))

    def update_state_diagram(self):
        state_frame = np.ones((200, 200, 3), np.uint8) * 0
        active_state_pos = {'P': (50, 50), 'F': (150, 50), 'U': (100, 150)}

        cv2.line(state_frame, (50, 50), (150, 50), (255, 255, 255), 1)
        cv2.line(state_frame, (150, 50), (100, 150), (255, 255, 255), 1)
        cv2.line(state_frame, (100, 150), (50, 50), (255, 255, 255), 1)

        cv2.circle(state_frame, (50, 50), 29, (0, 0, 0), -1)
        cv2.circle(state_frame, (150, 50), 29, (0, 0, 0), -1)
        cv2.circle(state_frame, (100, 150), 29, (0, 0, 0), -1)

        cv2.circle(state_frame, active_state_pos[self.active_state], 29, (100, 200, 0), -1)

        cv2.circle(state_frame, (50, 50), 30, (255, 255, 255), 1)
        cv2.putText(state_frame, 'Push', (30, 55), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(state_frame, (150, 50), 30, (255, 255, 255), 1)
        cv2.putText(state_frame, 'Find', (133, 55), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(state_frame, (100, 150), 30, (255, 255, 255), 1)
        cv2.putText(state_frame, 'Unwedge', (80, 152), cv2.FONT_HERSHEY_COMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("state_frame", state_frame)

    def get_feedback(self):
        for i in range(self.sensor_feedback_masks.shape[0]):
            self.sensor_feedback[2 * i] = np.any(
                (self.sensor_feedback_masks[i] + self.box_frame[:, :, 0]) == 150) or np.any(
                (self.sensor_feedback_masks[i] + self.neg_circle_frame[:, :, 0]) == 150)
            self.sensor_feedback[2 * i + 1] = np.any(
                (self.sensor_feedback_masks[i] + self.box_frame[:, :, 0]) == 200) or np.any(
                (self.sensor_feedback_masks[i] + self.neg_circle_frame[:, :, 0]) == 200)
        self.sensor_feedback[17] = self.stuck_flag

    def step(self, move, render=True):
        angle_change = self.move_options[move]
        self.facing_angle += angle_change
        self.active_state = 'F'
        if angle_change == 0:
            bot_center_x_t = int(
                self.bot_center_x + self.forward_step_unit * np.cos(np.deg2rad(self.facing_angle)))
            bot_center_y_t = int(
                self.bot_center_y + self.forward_step_unit * np.sin(np.deg2rad(self.facing_angle)))
            box_center_x_t = int(
                self.box_center_x + self.forward_step_unit * np.cos(np.deg2rad(self.facing_angle)))
            box_center_y_t = int(
                self.box_center_y + self.forward_step_unit * np.sin(np.deg2rad(self.facing_angle)))
            if self.enable_push:
                if (10 + self.bot_radius) < box_center_x_t < (self.frame_size[1] - 10 - self.bot_radius) and (
                            10 + self.bot_radius) < box_center_y_t < (
                                self.frame_size[
                                    0] - 10 - self.bot_radius):
                    self.box_center_x = box_center_x_t
                    self.box_center_y = box_center_y_t
                    self.bot_center_x = bot_center_x_t
                    self.bot_center_y = bot_center_y_t
                    self.stuck_flag = 0
                    self.active_state = 'P'
                else:
                    self.stuck_flag = 1
                    self.done = True
                    self.reward += 100*500000
                    self.active_state = 'U'

            elif (10 + self.bot_radius) < bot_center_x_t < (self.frame_size[1]- 10 - self.bot_radius) and (
                        10 + self.bot_radius) < bot_center_y_t < (self.frame_size[0] - 10 - self.bot_radius):
                self.bot_center_x = bot_center_x_t
                self.bot_center_y = bot_center_y_t
                self.stuck_flag = 0
            else:
                self.stuck_flag = 1
                self.active_state = 'U'

        if render:
            self.render_frame()
        self.get_feedback()
        self.update_reward()
        self.check_done_state()
        self.update_state_diagram()

        return self.sensor_feedback, self.reward, self.done

    def check_done_state(self):
        # cv2.imshow("added_bot_box", self.bot_mask[:, :, 0] + self.box_frame[:, :, 0])
        if np.any((self.bot_mask[:, :, 0] + self.box_frame[:, :, 0]) == 200):
            # self.done = True
            self.reward += 100
            y = (np.argmax((self.bot_mask[:, :, 0] + self.box_frame[:, :, 0])) // self.frame_size)[0]
            x = (np.argmax((self.bot_mask[:, :, 0] + self.box_frame[:, :, 0])) % self.frame_size)[0]
            # cv2.circle(self.frame, (x, int(self.frame_size[0]-y)), self.bot_radius//10, (250, 250, 250), -1)
            # cv2.imshow("asdf", self.frame)
            self.enable_push = True
            self.active_state = 'P'

            # print("************done*********************")
        elif np.any((self.bot_mask[:, :, 0] + self.neg_circle_frame[:, :, 0]) == 200):
            self.done = True
            self.reward += -100
            print("************Negative done*********************")

        # if self.bot_center_x == self.box_center_x and self.bot_center_y==self.bot_center_y:
        #     self.done = True
        #     self.reward = 100

    def update_reward(self):
        left_sensor_reward = np.sum(self.sensor_feedback[:4] * 1)
        forward_far_sensor_reward = np.sum(self.sensor_feedback[4:12][::2] * 2)
        forward_near_sensor_reward = np.sum(self.sensor_feedback[4:12][1::2] * 3)
        right_sensor_reward = np.sum(self.sensor_feedback[12:16] * 1)
        ir_sensor_reward = self.sensor_feedback[16] * 5
        stuck_reward = self.sensor_feedback[17] * (-5000)
        negative_reward = np.sum(np.logical_not(self.sensor_feedback)) * -1
        self.reward = left_sensor_reward + forward_far_sensor_reward + forward_near_sensor_reward + right_sensor_reward + ir_sensor_reward + stuck_reward +negative_reward


