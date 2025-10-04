#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "defs.h"

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static bool scrolled = false;
static double MouseDeltaX = 0.0;
static double MouseDeltaY = 0.0;

static double lastX = 0.0;
static double lastY = 0.0;
float zoom = 1.7f;
float theta = 1.0f, phi = 0.0f;
static bool bShouldSaveImage = false;
static bool bShouldResetCam = false;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;
GLFWwindow* window;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

// Forward declarations for window loop and interactivity
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures(int width, int height)
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initPBO(int width, int height)
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
    __debugbreak();
}

bool InitGLFW(int width, int height)
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetScrollCallback(window, scroll_callback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures(width, height);
    cudaGLSetGLDevice(0);
    // Clean up on program exit
    atexit(cleanupCuda);
    initPBO(width, height);
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui(GuiDataContainer* imguiData)
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Path Tracer Analytics");                  // Create a window called "Hello, world!" and append into it.
    ImGui::Checkbox("Debug Mode", &imguiData->isDebug);
	ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    bShouldSaveImage = ImGui::Button("Save Image");
    ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f * ImGui::GetIO().DeltaTime, ImGui::GetIO().Framerate);
    ImGui::End();
	ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------
void SaveImage(PathTraceInfo PTInfo, int iteration, std::string ImageName, const std::vector<glm::vec3>& ImageData);


int main(int argc, char** argv)
{
    PathTraceInfo PTInfo;
    std::string sceneFile = "";
    startTimeString = currentTimeString();

    PTInfo.x = 400;
    PTInfo.y = 400;
    PTInfo.frames = -1;
    PTInfo.depths = 16;
    PTInfo.ShowGUI = 2;

    for (int i = 1; i < argc; i+=2)
    {
        std::string key(argv[i]);

        if (key=="--x")
        {
            PTInfo.x = std::atoi(argv[i + 1]);
        }
        else if (key == "--y")
        {
            PTInfo.y = std::atoi(argv[i + 1]);
        }
        else if (key == "--xy")
        {
            PTInfo.x = std::atoi(argv[i + 1]);
            PTInfo.y = std::atoi(argv[i + 1]);
        }

        else if (key=="--frames")
        {
            PTInfo.frames = std::atoi(argv[i + 1]);
        }

        else if (key == "--depths")
        {
            PTInfo.depths = std::atoi(argv[i + 1]);
        }
        else if (key=="--gui")
        {
            PTInfo.ShowGUI = std::atoi(argv[i + 1]);
        }
        else if (i==argc-1)
        {
            sceneFile = argv[i];
        }
    	else
        {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            printf("Usage: %s SCENEFILE.json / SCENEFILE.gltf\n", argv[0]);
            return 1;
        }
    }

    if (sceneFile.empty())
    {
        printf("No scene file given! \n");
        return 1;
    }

    std::cout << "SirEnri's Path Tracer [MIT License]" << std::endl;
    std::cout << "Scene File: " << sceneFile << std::endl;
    std::cout << "Viewport Size: "<<PTInfo.x<<", "<<PTInfo.y << std::endl;
    if (PTInfo.frames<0)
    {
        std::cout << "Total Frames: " << "Unlimited" << std::endl;
    }else
    {
        std::cout << "Total Frames: " << PTInfo.frames << std::endl;
    }
    std::cout << "Total Depths: " << PTInfo.depths << std::endl;
    if (PTInfo.ShowGUI==0)
    {
        std::cout << "Running in Commandlet Mode" << std::endl;
    }else
    {
        std::cout << "Running in GUI Mode. Please use Commandlet (--gui 0) to profile in NSight Compute" << std::endl;
    }

    // Load scene file
    Scene* SceneInstance = new Scene(PTInfo);
    if (sceneFile.find(".json")!=std::string::npos)
    {
        SceneInstance->ReadJSON(sceneFile);
    }
    else if (sceneFile.find(".gltf") != std::string::npos)
    {
        SceneInstance->ReadGLTF(sceneFile);
    }
	else
    {
        printf("Unsupported file type: %s\n", sceneFile.c_str());
        return 1;
    }
    SceneInstance->PostLoad();
    SceneInstance->CreateDefaultLight();
    SceneInstance->CreateDefaultFloor();

    // Set up camera stuff from loaded path tracer settings
    Camera& cam = SceneInstance->state.camera;

    // Initialize CUDA and GL components
    if (PTInfo.ShowGUI>0)
    {
        InitGLFW(PTInfo.x, PTInfo.y);
    }

    // Create path tracer and scene data
    PTEngine Engine(PTInfo);
    Engine.Init();

    SceneInstance->CreateRenderProxyForAll();
    SceneInstance->CenterCamera();
    cam.SetCamera(phi, theta, zoom, 0., 0.);
    // GLFW main loop
    if (PTInfo.ShowGUI > 0) {
        while (!glfwWindowShouldClose(window) && (PTInfo.frames<0 || SceneInstance->state.frames < PTInfo.frames))
        {
            glfwPollEvents();

            bool camMoved = scrolled;
            float transX = 0.f, transY = 0.f;
            if (leftMousePressed && (abs(MouseDeltaX) > 0.0 || abs(MouseDeltaY) > 0.0))
            {
                // compute new camera parameters
                phi -= MouseDeltaX / PTInfo.x;
                theta -= MouseDeltaY / PTInfo.y;
                theta = std::fmax(0.001f, std::fmin(theta, PI));
                camMoved = true;
            }
            if (rightMousePressed && (abs(MouseDeltaX) > 0.0 || abs(MouseDeltaY) > 0.0))
            {
                transX = MouseDeltaX;
                transY = MouseDeltaY;
                camMoved = true;
            }
            if (bShouldResetCam)
            {
                bShouldResetCam = false;
                cam.Reset();
                camMoved = true;
            }
            if (camMoved)
            {
                camMoved = false;
                cam.SetCamera(phi, theta, zoom, transX, transY);
                Engine.ClearBuffers();
                SceneInstance->state.frames = 0;
            }
            scrolled = false;

            // Map OpenGL buffer object for writing from CUDA on a single GPU
            // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

            uchar4* pbo_dptr = NULL;
            cudaGLMapBufferObject((void**)&pbo_dptr, pbo);
            Engine.RenderResource.pbo = pbo_dptr;
            // execute the kernel
            Engine.Tick(SceneInstance);

            // unmap buffer object
            cudaGLUnmapBufferObject(pbo);

            std::string title = "SirEnri's CUDA Path Tracer";
            glfwSetWindowTitle(window, title.c_str());
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glBindTexture(GL_TEXTURE_2D, displayImage);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, PTInfo.x, PTInfo.y, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            glClear(GL_COLOR_BUFFER_BIT);

            // Binding GL_PIXEL_UNPACK_BUFFER back to default
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            // VAO, shader program, and texture already bound
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

            if (PTInfo.ShowGUI==2)
            {
                // Render ImGui Stuff
                RenderImGui(&Engine.GuiData);
            }
            if (bShouldSaveImage)
            {
                bShouldSaveImage = false;
                std::cout << "Saving image at frame " << SceneInstance->state.frames << std::endl;
                SaveImage(PTInfo, SceneInstance->state.frames, "PT", SceneInstance->state.image);
            }
            glfwSwapBuffers(window);
            SceneInstance->state.frames++;
        }
    }
    else {
        while (SceneInstance->state.frames < PTInfo.frames)
        {
            Engine.RenderResource.pbo = nullptr;
            // execute the kernel
            Engine.Tick(SceneInstance);
            SceneInstance->state.frames++;
            if (SceneInstance->state.frames%20)
            {
                SaveImage(PTInfo, SceneInstance->state.frames, "PT", SceneInstance->state.image);
            }
        }
    }
    SaveImage(PTInfo, SceneInstance->state.frames, "PT_Final", SceneInstance->state.image);
    Engine.Destroy();
    cudaDeviceReset();

#if !COMMANDLET
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
#endif

    exit(EXIT_SUCCESS);
}

void SaveImage(PathTraceInfo PTInfo, int frames, std::string ImageName, const std::vector<glm::vec3>& ImageData)
{
    float samples = (float)frames;
    // output image file
    Image img(PTInfo.x, PTInfo.y);

    for (int x = 0; x < PTInfo.x; x++)
    {
        for (int y = 0; y < PTInfo.y; y++)
        {
            int index = x + (y * PTInfo.x);
            glm::vec3 pix = ImageData[index];
            img.setPixel(PTInfo.x - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = ImageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << ".frames" << frames;
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            bShouldSaveImage = true;
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_S:
            bShouldSaveImage = true;
            break;
        case GLFW_KEY_SPACE:
            bShouldResetCam = true;
            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    scrolled = true;
    if (yoffset > 0)
    {
        zoom -= 0.1f;
        zoom = std::fmax(0.1f, zoom);
    }
    else if (yoffset < 0)
    {
        zoom += 0.1f;
	}
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    MouseDeltaX = xpos - lastX;
    MouseDeltaY = ypos - lastY;
    lastX = xpos;
    lastY = ypos;
}
