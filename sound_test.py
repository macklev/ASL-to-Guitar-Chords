import pygame

pygame.mixer.init()
pygame.mixer.music.load("sounds/G.wav")  # change if needed
pygame.mixer.music.play()

input("Press Enter to stop...")