@echo off
setlocal

set "ROOT=%USERPROFILE%\.conan2"
set "TARGET=windeployqt.exe"

echo [0/6] Searching for %TARGET% in %ROOT% ...

set "WINDEPLOYQT="

for /r "%ROOT%" %%f in (*%TARGET%) do (
    if /i "%%~nxf"=="%TARGET%" (
        set "WINDEPLOYQT=%%~f"
        goto :found
    )
)

echo [ERROR] %TARGET% not found in %ROOT%
exit /b 1

:found
echo Found windeployqt: %WINDEPLOYQT%
echo.

echo [1/6] Installing Python dependencies...
poetry install --no-root || goto :error

echo [2/6] Installing Conan dependencies...
poetry run conan install . -pr:h=profiles/windows_msvc17.txt -pr:b=profiles/windows_msvc17.txt -s build_type=Release --build=missing || goto :error

echo [3/6] Generating CMake project...
cmake -S . -B build/Release -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release || goto :error

echo [4/6] Building project...
cmake --build build/Release --config Release -j || goto :error

echo [5/6] Deploying Qt libraries...
"%WINDEPLOYQT%" build\Release\bin\Release\nal.exe --dir build\Release\bin\Release --release || goto :error

echo [6/6] Done!
echo === BUILD SUCCESSFUL ===
exit /b 0

:error
echo === BUILD FAILED ===
exit /b 1




