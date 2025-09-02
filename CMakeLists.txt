cmake_minimum_required(VERSION 3.23)

# Корневой проект без указания языков — он лишь "сценарист"
project(NAL LANGUAGES NONE)

# Пример шага для Python (по желанию): подготовить окружение Poetry
# Это "фиктивная" цель, которую можно заставить выполняться раньше GUI.
# add_custom_target(py_setup ALL
#   COMMAND poetry install --no-interaction --no-ansi
#   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#   COMMENT "Poetry: install deps"
#   VERBATIM
# )

# Подключаем C++/Qt часть отдельным подпроектом
add_subdirectory(gui)
