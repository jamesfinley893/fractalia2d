#!/bin/bash

# Create shaders directories if they don't exist
mkdir -p build/shaders/compiled
mkdir -p src/shaders/compiled

# Compile vertex shader
glslangValidator -V src/shaders/vertex.vert -o src/shaders/compiled/vertex.vert.spv
cp src/shaders/compiled/vertex.vert.spv build/shaders/

# Compile fragment shader  
glslangValidator -V src/shaders/fragment.frag -o src/shaders/compiled/fragment.frag.spv
cp src/shaders/compiled/fragment.frag.spv build/shaders/

# Compile compute shader (random walk movement)
glslangValidator -V src/shaders/movement_random.comp -o src/shaders/compiled/movement_random.comp.spv
cp src/shaders/compiled/movement_random.comp.spv build/shaders/

# Compile compute shader (physics)
glslangValidator -V src/shaders/physics.comp -o src/shaders/compiled/physics.comp.spv
cp src/shaders/compiled/physics.comp.spv build/shaders/

# Compile compute shader (PBD soft bodies)
glslangValidator -V src/shaders/pbd.comp -o src/shaders/compiled/pbd.comp.spv
cp src/shaders/compiled/pbd.comp.spv build/shaders/

# Export shaders to Windows build folder
WINDOWS_DEST="/mnt/f/Projects/Fractalia2/build/shaders"
if mkdir -p "$WINDOWS_DEST" 2>/dev/null; then
    cp src/shaders/compiled/*.spv "$WINDOWS_DEST/"
    echo "Shaders exported to Windows build folder: $WINDOWS_DEST"
else
    echo "Warning: Could not export to Windows build folder (path may not exist)"
fi

echo "Shaders compiled successfully!"
