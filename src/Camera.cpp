#include "Camera.h"
#include <algorithm>
#include <iostream>

Camera::Camera()
    : m_posX(0.0f), m_posY(0.0f), m_posZ(3.0f)
    , m_yaw(-3.14159265f)  // Facing -Z (towards origin from +Z)
    , m_pitch(0.0f)
    , m_speed(2.0f)
    , m_sensitivity(0.003f)
    , m_speedMultiplier(0.1f)
    , m_freeCameraMode(false)
{
    UpdateVectors();
}

void Camera::Initialize(float x, float y, float z) {
    m_posX = x;
    m_posY = y;
    m_posZ = z;
    
    // Calculate initial yaw to face origin
    float dx = -x;
    float dz = -z;
    m_yaw = atan2f(dx, dz);
    m_pitch = 0.0f;
    
    UpdateVectors();
}

void Camera::UpdateVectors() {
    // Calculate front vector from yaw and pitch
    m_frontX = sinf(m_yaw) * cosf(m_pitch);
    m_frontY = sinf(m_pitch);
    m_frontZ = cosf(m_yaw) * cosf(m_pitch);
    
    // Normalize front
    float frontLen = sqrtf(m_frontX * m_frontX + m_frontY * m_frontY + m_frontZ * m_frontZ);
    if (frontLen > 0.0001f) {
        m_frontX /= frontLen;
        m_frontY /= frontLen;
        m_frontZ /= frontLen;
    }
    
    // Calculate right vector (cross product of front and world up)
    m_rightX = m_frontY * WORLD_UP_Z - m_frontZ * WORLD_UP_Y;
    m_rightY = m_frontZ * WORLD_UP_X - m_frontX * WORLD_UP_Z;
    m_rightZ = m_frontX * WORLD_UP_Y - m_frontY * WORLD_UP_X;
    
    // Normalize right
    float rightLen = sqrtf(m_rightX * m_rightX + m_rightY * m_rightY + m_rightZ * m_rightZ);
    if (rightLen > 0.0001f) {
        m_rightX /= rightLen;
        m_rightY /= rightLen;
        m_rightZ /= rightLen;
    }
    
    // Calculate up vector (cross product of right and front)
    m_upX = m_rightY * m_frontZ - m_rightZ * m_frontY;
    m_upY = m_rightZ * m_frontX - m_rightX * m_frontZ;
    m_upZ = m_rightX * m_frontY - m_rightY * m_frontX;
    
    // Normalize up
    float upLen = sqrtf(m_upX * m_upX + m_upY * m_upY + m_upZ * m_upZ);
    if (upLen > 0.0001f) {
        m_upX /= upLen;
        m_upY /= upLen;
        m_upZ /= upLen;
    }
}

void Camera::ProcessKeyboard(GLFWwindow* window, float deltaTime) {
    if (!m_freeCameraMode) return;
    
    // Check if shift is held for speed boost
    float speed = m_speed;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
        speed *= m_speedMultiplier;
    }
    
    float velocity = speed * deltaTime;
    
    // W/S - Forward/Backward
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        m_posX += m_frontX * velocity;
        m_posY += m_frontY * velocity;
        m_posZ += m_frontZ * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        m_posX -= m_frontX * velocity;
        m_posY -= m_frontY * velocity;
        m_posZ -= m_frontZ * velocity;
    }
    
    // A/D - Strafe Left/Right
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        m_posX -= m_rightX * velocity;
        m_posY -= m_rightY * velocity;
        m_posZ -= m_rightZ * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        m_posX += m_rightX * velocity;
        m_posY += m_rightY * velocity;
        m_posZ += m_rightZ * velocity;
    }
}

void Camera::ProcessMouseMovement(float xOffset, float yOffset) {
    if (!m_freeCameraMode) return;
    
    m_yaw -= xOffset * m_sensitivity;
    m_pitch -= yOffset * m_sensitivity;  // Inverted Y
    
    // Clamp pitch to avoid gimbal lock
    m_pitch = std::max(-PITCH_LIMIT, std::min(PITCH_LIMIT, m_pitch));
    
    // Keep yaw in reasonable range
    if (m_yaw > 6.28318530718f) m_yaw -= 6.28318530718f;
    if (m_yaw < -6.28318530718f) m_yaw += 6.28318530718f;
    
    UpdateVectors();
}

void Camera::AdjustSpeed(float delta) {
    m_speed += delta;
    m_speed = std::max(0.1f, std::min(50.0f, m_speed));  // Clamp between 0.1 and 50
    std::cout << "Camera speed: " << m_speed << std::endl;
}

void Camera::GetViewMatrix(float outMatrix[16]) const {
    // Build view matrix (column-major for OpenGL)
    // This is equivalent to gluLookAt
    
    // Right vector
    outMatrix[0] = m_rightX;
    outMatrix[1] = m_upX;
    outMatrix[2] = -m_frontX;
    outMatrix[3] = 0.0f;
    
    outMatrix[4] = m_rightY;
    outMatrix[5] = m_upY;
    outMatrix[6] = -m_frontY;
    outMatrix[7] = 0.0f;
    
    outMatrix[8] = m_rightZ;
    outMatrix[9] = m_upZ;
    outMatrix[10] = -m_frontZ;
    outMatrix[11] = 0.0f;
    
    // Translation (dot products)
    outMatrix[12] = -(m_rightX * m_posX + m_rightY * m_posY + m_rightZ * m_posZ);
    outMatrix[13] = -(m_upX * m_posX + m_upY * m_posY + m_upZ * m_posZ);
    outMatrix[14] = -(-m_frontX * m_posX + -m_frontY * m_posY + -m_frontZ * m_posZ);
    outMatrix[15] = 1.0f;
}

void Camera::GetPosition(float& x, float& y, float& z) const {
    x = m_posX;
    y = m_posY;
    z = m_posZ;
}

void Camera::SetPosition(float x, float y, float z) {
    m_posX = x;
    m_posY = y;
    m_posZ = z;
}

void Camera::SetYaw(float yaw) {
    m_yaw = yaw;
    UpdateVectors();
}

void Camera::SetPitch(float pitch) {
    m_pitch = std::max(-PITCH_LIMIT, std::min(PITCH_LIMIT, pitch));
    UpdateVectors();
}

void Camera::SetSpeed(float speed) {
    m_speed = std::max(0.1f, std::min(50.0f, speed));
}

void Camera::SetSensitivity(float sensitivity) {
    m_sensitivity = std::max(0.0001f, std::min(0.01f, sensitivity));
}
