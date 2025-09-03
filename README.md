 Для того, чтобы собрать всю эту богодельню, вам понадобиться:

 Poetry
 CMake
 Ninja
 Компилятор для C++ в зависимости от вашей платформы

 Далее путем следования командам ниже, вы сможете получить готовый проект:

```bash
poetry install --no-root
poetry run conan install . -pr:h=profiles/linux_gcc13.txt -pr:b=profiles/linux_gcc13.txt -s build_type=Release --build=missing
cmake -S . -B build/Release -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release -j
```

Прошу учесть, что в примере выше указано profiles/linux_gcc13.txt - будьте внимательны и выберите нужный вам вариант!