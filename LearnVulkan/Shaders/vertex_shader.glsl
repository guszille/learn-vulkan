#version 450

layout(binding = 0) uniform UniformBufferObject
{
    mat4 model;
    mat4 view;
    mat4 projection;
} UBO;

layout(location = 0) in  vec2 inPosition;
layout(location = 1) in  vec3 inColor;

layout(location = 0) out vec3 fragmentColor;

void main()
{
    gl_Position = UBO.projection * UBO.view * UBO.model * vec4(inPosition, 0.0, 1.0);
    fragmentColor = inColor;
}