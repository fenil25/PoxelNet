import numpy as np
import math
import random


class CoordinateTransformation:
    def __init__(self, scale_range=(0.9, 1.1), 
                 rot_range = {"X": (-30, 31), "Y": (-30, 31), "Z": (-30, 31)}, 
                 trans=0.25, jitter=0.025, clip=0.05):
        self.scale_range = scale_range
        self.rot_range = rot_range
        self.trans = trans
        self.jitter = jitter
        self.clip = clip

    def X_rotation(self, points, degree):
        theta = ((math.pi)/180)*degree
        Rx = np.matrix([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
        return np.matmul(points, Rx)

    def Y_rotation(self, points, degree):
        theta = ((math.pi)/180)*degree
        Ry = np.matrix([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
        return np.matmul(points, Ry)

    def Z_rotation(self, points, degree):
        theta = ((math.pi)/180)*degree
        Rz = np.matrix([[math.cos(theta), -math.sin(theta), 0 ], [math.sin(theta), math.cos(theta), 0],[0, 0, 1]])
        return np.matmul(points, Rz)

    def translate(self, points, translate):
        return points + translate

    def scale(self, points, scaling_factor):
        return points * scaling_factor

    def jittering(self, points, jitter_by):
        return points + np.clip(self.jitter * jitter_by, -self.clip, self.clip)

    def apply_transformation(self, points):
        if random.random() < 0.5:
            X_rot = np.random.randint(self.rot_range["X"][0], self.rot_range["X"][1])
            points = self.X_rotation(points, X_rot)
        if random.random() < 0.5:
            Y_rot = np.random.randint(self.rot_range["Y"][0], self.rot_range["Y"][1])
            points = self.Y_rotation(points, Y_rot)
        if random.random() < 0.5:
            Z_rot = np.random.randint(self.rot_range["Y"][0], self.rot_range["Z"][1])
            points = self.Z_rotation(points, Z_rot)

        points = np.array(points)

        if random.random() < 0.9:
            translation = np.random.uniform(-self.trans, self.trans, size=(1, 3))
            points = self.translate(points, translation)
        if random.random() < 0.9:
            scaling = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(1, 3))
            points = self.scale(points, scaling)

        if random.random() < 0.6:
            jitter_by = (np.random.rand(*points.shape) - 0.5)
            points = self.jittering(points, jitter_by)

        return points