#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _APPLE_
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#define DEG2RAD (M_PI / 180.0f)

typedef float mat4[4][4];
typedef float vec4[4];

static vec4 transformedCube[8];  // after transformations
static vec4 baseCube[8];         // original cube vertices
static mat4 modelMatrix;         // keeps track of transformations

// viewing params
static int win_w = 800, win_h = 600;
static float viewRotX = 20.0f, viewRotY = -30.0f;
static int lastX = -1, lastY = -1, dragging = 0;

// ---------- Matrix Utilities ----------
void makeIdentity(mat4 m) {
    memset(m, 0, sizeof(mat4));
    for (int i = 0; i < 4; i++) m[i][i] = 1.0f;
}

void multiplyMat4(const mat4 a, const mat4 b, mat4 out) {
    mat4 tmp;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            tmp[r][c] = 0.0f;
            for (int k = 0; k < 4; k++) tmp[r][c] += a[r][k] * b[k][c];
        }
    memcpy(out, tmp, sizeof(mat4));
}

void multiplyVec4(const mat4 m, const vec4 v, vec4 out) {
    for (int r = 0; r < 4; r++)
        out[r] = m[r][0]*v[0] + m[r][1]*v[1] + m[r][2]*v[2] + m[r][3]*v[3];
}