import numpy as np

import OpenGL.GL
import OpenGL.GLUT
import OpenGL.GLU

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

verts = (
 (1, -1, -1),
 (1, 1, -1),
 (-1, 1, -1),
 (-1, -1, -1),
 (1, -1, 1),
 (1, 1, 1),
 (-1, -1, 1),
 (-1, 1, 1)
 )

surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
    )

def cube(pos=(0,0,0),scale=(1.0,1.0,1.0),col=(1,0,0,0.1)):
    colours = (col,col,col,col,col,col,col,col)

    glBegin(GL_QUADS)
    for surface in surfaces:
      x = 0
      for vertex in surface:
        x += 1
        glColor4fv(colours[x])
        vert = [0 for v in verts[vertex]]
        vert[0] = (verts[vertex][0] * scale[0]) + pos[0]
        vert[1] = (verts[vertex][1] * scale[1]) + pos[1]
        vert[2] = (verts[vertex][2] * scale[2]) + pos[2]
        glVertex3fv(vert)

    glEnd()


def buildGrid(res=(10,10,10), grid_centre=(0,0,0), grid_size=(2.0,2.0,2.0)):
    grid_vals = [0 for a in range(res[0]*res[1]*res[2])]
    grid_loc = [(0.0,0.0,0.0) for a in range(res[0]*res[1]*res[2])]

    cell_count = 0
    for x in range(res[0]):
        for y in range(res[1]):
            for z in range(res[2]):
                grid_vals[cell_count] = 0.1
                grid_loc[cell_count] = (grid_centre[0] - (grid_size[0]/2.0) + x*(grid_size[0]/res[0]),grid_centre[1]- (grid_size[1]/2.0) + y*(grid_size[1]/res[1]),grid_centre[2]- (grid_size[2]/2.0) + z*(grid_size[2]/res[2]))
                cell_count += 1

    return grid_vals, grid_loc
    
def main():
    pygame.init()
    pygame.display.set_caption('Visualiser')
    display = (800, 800)
    pygame.display.set_mode(display,DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glEnable(GL_BLEND)
    glTranslatef(0.0,0.0,-5)

    glRotatef(0, 0, 0, 0)

    grid_vals, grid_loc = buildGrid()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        glRotatef(1, 3, 1, 1)
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        for c in range(len(grid_vals)):
            cube(pos=grid_loc[c], scale=(0.2,0.2,0.2), col=(1,0,0,grid_vals[c]))
        pygame.display.flip()
        pygame.time.wait(10)
main()