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
// ---------- Transformation Generators ----------
void makeTranslation(float tx, float ty, float tz, mat4 out) {
    makeIdentity(out);
    out[0][3] = tx; out[1][3] = ty; out[2][3] = tz;
}

void makeScale(float sx, float sy, float sz, mat4 out) {
    makeIdentity(out);
    out[0][0] = sx; out[1][1] = sy; out[2][2] = sz;
}

void makeRotationX(float deg, mat4 out) {
    makeIdentity(out);
    float a = deg * DEG2RAD;
    out[1][1] = cosf(a); out[1][2] = -sinf(a);
    out[2][1] = sinf(a); out[2][2] = cosf(a);
}

void makeRotationY(float deg, mat4 out) {
    makeIdentity(out);
    float a = deg * DEG2RAD;
    out[0][0] = cosf(a); out[0][2] = sinf(a);
    out[2][0] = -sinf(a); out[2][2] = cosf(a);
}

void makeRotationZ(float deg, mat4 out) {
    makeIdentity(out);
    float a = deg * DEG2RAD;
    out[0][0] = cosf(a); out[0][1] = -sinf(a);
    out[1][0] = sinf(a); out[1][1] = cosf(a);
}

void makeReflection(char axis, mat4 out) {
    makeIdentity(out);
    if (axis == 'x') out[0][0] = -1;
    if (axis == 'y') out[1][1] = -1;
    if (axis == 'z') out[2][2] = -1;
}

void makeShear(int type, float sh, mat4 out) {
    makeIdentity(out);
    switch (type) {
        case 1: out[0][1] = sh; break; // X += sh*Y
        case 2: out[0][2] = sh; break; // X += sh*Z
        case 3: out[1][0] = sh; break; // Y += sh*X
        case 4: out[1][2] = sh; break; // Y += sh*Z
        case 5: out[2][0] = sh; break; // Z += sh*X
        case 6: out[2][1] = sh; break; // Z += sh*Y
    }
}

void applyTransform(const mat4 M) {
    multiplyMat4(M, modelMatrix, modelMatrix);
    for (int i = 0; i < 8; i++)
        multiplyVec4(modelMatrix, baseCube[i], transformedCube[i]);
}
// ---------- Cube Setup ----------
void initCube() {
    float s = 0.6f; // half-size
    vec4 verts[8] = {
        {-s,-s,-s,1}, { s,-s,-s,1}, { s, s,-s,1}, {-s, s,-s,1},
        {-s,-s, s,1}, { s,-s, s,1}, { s, s, s,1}, {-s, s, s,1}
    };
    for (int i = 0; i < 8; i++) memcpy(baseCube[i], verts[i], sizeof(vec4));
    makeIdentity(modelMatrix);
    for (int i = 0; i < 8; i++) multiplyVec4(modelMatrix, baseCube[i], transformedCube[i]);
}

// ---------- Drawing ----------
void drawAxes(float len) {
    glBegin(GL_LINES);
      glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(len,0,0);
      glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,len,0);
      glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,len);
    glEnd();
}

void drawCube() {
    int faces[6][4] = {
        {0,1,2,3}, {4,5,6,7}, // back & front
        {0,4,7,3}, {1,5,6,2}, // left & right
        {3,2,6,7}, {0,1,5,4}  // top & bottom
    };

    float colors[6][3] = {
        {0.9f,0.4f,0.4f}, {0.4f,0.9f,0.4f}, {0.4f,0.4f,0.9f},
        {0.9f,0.9f,0.4f}, {0.9f,0.6f,0.9f}, {0.4f,0.9f,0.9f}
    };

    for (int f = 0; f < 6; f++) {
        glColor3fv(colors[f]);
        glBegin(GL_QUADS);
        for (int v = 0; v < 4; v++) {
            int idx = faces[f][v];
            glVertex3f(transformedCube[idx][0],
                       transformedCube[idx][1],
                       transformedCube[idx][2]);
        }
        glEnd();
    }

    glColor3f(0,0,0);
    glBegin(GL_LINES);
    int edges[12][2] = {
        {0,1},{1,2},{2,3},{3,0},
        {4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7}
    };
    for (int i=0;i<12;i++) {
        int a=edges[i][0], b=edges[i][1];
        glVertex3f(transformedCube[a][0],transformedCube[a][1],transformedCube[a][2]);
        glVertex3f(transformedCube[b][0],transformedCube[b][1],transformedCube[b][2]);
    }
    glEnd();
}
