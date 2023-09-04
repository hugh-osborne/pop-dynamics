import numpy as np

import OpenGL.GL
import OpenGL.GLUT
import OpenGL.GLU

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

class Visualiser:
    def __init__(self):
        self.verts = (
         (1, -1, -1),
         (1, 1, -1),
         (-1, 1, -1),
         (-1, -1, -1),
         (1, -1, 1),
         (1, 1, 1),
         (-1, -1, 1),
         (-1, 1, 1)
         )

        self.surfaces = (
            (0,1,2,3),
            (3,2,7,6),
            (6,7,5,4),
            (4,5,1,0),
            (1,5,7,2),
            (4,0,3,6)
            )

    def cube(self, pos=(0,0,0),scale=(1.0,1.0,1.0),col=(1,0,0,0.1)):
        colours = (col,col,col,col,col,col,col,col)

        glBegin(GL_QUADS)
        for surface in self.surfaces:
          x = 0
          for vertex in surface:
            x += 1
            glColor4fv(colours[x])
            vert = [0 for v in self.verts[vertex]]
            vert[0] = (self.verts[vertex][0] * scale[0]) + pos[0]
            vert[1] = (self.verts[vertex][1] * scale[1]) + pos[1]
            vert[2] = (self.verts[vertex][2] * scale[2]) + pos[2]
            glVertex3fv(vert)

        glEnd()

    def drawCell(self, cell_coords, cell_mass, origin_location=(0.0,0.0,0.0), max_size=(2.0,2.0,2.0), max_res=(10,10,10)):
        widths = (max_size[0]/max_res[0], max_size[1]/max_res[1], max_size[2]/max_res[2])
        pos = (origin_location[0] - (max_size[0]/2.0) + ((cell_coords[0]+0.5)*widths[0]), origin_location[1] - (max_size[1]/2.0) + ((cell_coords[1]+0.5)*widths[1]), origin_location[2] - (max_size[2]/2.0) + ((cell_coords[2]+0.5)*widths[2]))
        self.cube(pos, scale=widths, col=(cell_mass,0,0,0.1))

    def setupVisuliser(self, display_size=(800,800)):
        pygame.init()
        pygame.display.set_caption('Visualiser')
        display = display_size
        pygame.display.set_mode(display,DOUBLEBUF|OPENGL)

        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
        glEnable(GL_BLEND)
        glTranslatef(0.0,0.0,-5)
        glRotatef(0, 0, 0, 0)

    def beginRendering(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        glRotatef(1, 3, 1, 1)
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)


    def endRendering(self):
        pygame.display.flip()
        pygame.time.wait(10)
