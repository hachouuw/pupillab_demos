from pymouse import PyMouse
import pygame
import numpy as np
# x, y = pygame.mouse.get_pos()
cursor_coor = np.array(pygame.mouse.get_pos(), dtype=np.float)
print(cursor_coor)