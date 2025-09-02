cmake_minimum_required(VERSION 3.23)

# Корневой проект без указания языков — он лишь "сценарист"
project(NAL LANGUAGES NONE)

# Все бинарники (бэк и фронт) будут лежать в одной директории
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Сборка Python-бэка через Poetry и PyInstaller
add_custom_target(backend ALL
    COMMAND poetry run pyinstaller --onefile ${CMAKE_SOURCE_DIR}/src/backend.py --distpath ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building Python backend"
)

# Подключаем C++/Qt часть отдельным подпроектом
add_subdirectory(gui)

# Гарантируем, что сначала соберётся бэк
add_dependencies(nal backend)
