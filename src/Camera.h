#pragma once

#include <GLFW/glfw3.h>
#include <cmath>

class Camera {
public:
    Camera();
    
    // Initialize with starting position
    void Initialize(float x, float y, float z);
    
    // Process keyboard input for movement (call each frame)
    void ProcessKeyboard(GLFWwindow* window, float deltaTime);
    
    // Process mouse movement for rotation (only when Alt is held)
    void ProcessMouseMovement(float xOffset, float yOffset);
    
    // Adjust movement speed with +/- keys
    void AdjustSpeed(float delta);
    
    // Get the view matrix (column-major for OpenGL)
    void GetViewMatrix(float outMatrix[16]) const;
    
    // Get camera position
    void GetPosition(float& x, float& y, float& z) const;
    
    // Setters
    void SetPosition(float x, float y, float z);
    void SetYaw(float yaw);
    void SetPitch(float pitch);
    void SetSpeed(float speed);
    void SetSensitivity(float sensitivity);
    
    // Getters
    float GetYaw() const { return m_yaw; }
    float GetPitch() const { return m_pitch; }
    float GetSpeed() const { return m_speed; }
    float GetSensitivity() const { return m_sensitivity; }
    
    // Toggle free camera mode vs orbit mode
    void SetFreeCameraMode(bool enabled) { m_freeCameraMode = enabled; }
    bool IsFreeCameraMode() const { return m_freeCameraMode; }

private:
    void UpdateVectors();
    
    // Position
    float m_posX, m_posY, m_posZ;
    
    // Euler angles (in radians)
    float m_yaw;    // Rotation around Y axis
    float m_pitch;  // Rotation around X axis
    
    // Camera vectors (derived from yaw/pitch)
    float m_frontX, m_frontY, m_frontZ;
    float m_rightX, m_rightY, m_rightZ;
    float m_upX, m_upY, m_upZ;
    
    // World up vector
    static constexpr float WORLD_UP_X = 0.0f;
    static constexpr float WORLD_UP_Y = 1.0f;
    static constexpr float WORLD_UP_Z = 0.0f;
    
    // Movement settings
    float m_speed;          // Base movement speed (units per second)
    float m_sensitivity;    // Mouse sensitivity
    float m_speedMultiplier; // Shift key multiplier
    
    // Mode
    bool m_freeCameraMode;  // True = free camera, False = orbit mode
    
    // Constraints
    static constexpr float PITCH_LIMIT = 1.5533f; // ~89 degrees in radians
};
