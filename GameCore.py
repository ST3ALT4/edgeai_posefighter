import pygame
import cv2 as cv
import numpy as np

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

#opencv setup
cam = cv.VideoCapture(0);
if not cam.isOpened():
    print("Cannot open camera")
    exit()

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    ret, frame = cam.read()
    if not ret:
        print("Can't recive frame")
        break

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB);

    frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))

    # fill the screen with a color to wipe away anything from last frame
    screen.blit(frame_surface, (0, 0))

    # RENDER YOUR GAME HERE

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()
