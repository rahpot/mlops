# Что сделано

Реализовал `tensor_ops.mac(a, b, c)` — считает `a * b + c` поэлементно для трёх 3D-массивов.

# TODO 1: ядро
`starter/cpp/kernel.cpp`:
Реализован код для поэлементного сложения: out[i] = a[i] * b[i] + c[i].


# TODO 2: биндинги
`starter/cpp/bindings.cpp`:
Проверяет массивы на правильность:
- dtype сравнивается через `dtype().equal(py::dtype::of<double>())`.
- `ndim() == 3`.
- сравниваются формы по осям.
- флаг `c_style` и `flags.aligned`.

Неверный dtype и не-массив дают `TypeError`, неверная форма или размещение — `ValueError`.
У всех трёх аргументов стоит `noconvert()`.

Результат — новый `py::array_t<double>` той же формы, имеющий свой буфер. Входы только читаются,
read-only массивы допустимы.


# TODO 3: сборка
В `CMakeLists.txt` добавил `pybind11_add_module(_core cpp/bindings.cpp cpp/kernel.cpp)`,
`cxx_std_17` и install в `tensor_ops`. В `pyproject.toml` дописал описание и
`dependencies = ["numpy==2.5.3"]`.

Сборка и проверка из чистого клона:
```bash
cd homework/01-bindings
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m build starter --outdir starter/dist
python -m pip install --force-reinstall starter/dist/mlops_tensor_ops-0.1.0-cp313-cp313-linux_x86_64.whl
python -m pytest tests student_tests -q
```

`python -m build` сначала собирает sdist и уже из него wheel, так что попутно проверяется,
что в архив попали оба `.cpp` и заголовок.


# Проверка wheel в чистом env
Собранный wheel:
```
mlops_tensor_ops-0.1.0-cp313-cp313-linux_x86_64.whl
sha256 15f2686b30775e5ca968fed4860bbb209223a805725c8ab3ebaeff76a8eb4fe6
```

Сделал новый venv, поставил туда только wheel и pytest, тесты скопировал в отдельную папку:

```bash
hw_root="$PWD"
hw_receiver=$(mktemp -d)
python3.13 -m venv "$hw_receiver/venv"
env -u PYTHONPATH "$hw_receiver/venv/bin/python" -m pip install \
  pytest==9.1.1 "$hw_root"/starter/dist/mlops_tensor_ops-0.1.0-cp313-cp313-linux_x86_64.whl
cp -R tests student_tests "$hw_receiver/"
cd "$hw_receiver"
env -u PYTHONPATH "$hw_receiver/venv/bin/python" -m pip check
env -u PYTHONPATH "$hw_receiver/venv/bin/python" -m pytest tests student_tests -q
env -u PYTHONPATH "$hw_receiver/venv/bin/python" -c 'import tensor_ops._core as m; print(m.__file__)'
env -u PYTHONPATH "$hw_receiver/venv/bin/python" -m pip list
```

Вывод:
```
No broken requirements found.

43 passed in 0.08s

/tmp/tmp.lk9T1E0gwk/venv/lib/python3.13/site-packages/tensor_ops/_core.cpython-313-x86_64-linux-gnu.so

Package          Version
---------------- -------
iniconfig        2.3.0
mlops-tensor-ops 0.1.0
numpy            2.5.3
packaging        26.3
pip              26.2.1
pluggy           1.6.0
Pygments         2.21.0
pytest           9.1.1
```

NumPy подтянулся сам, `build`, `scikit-build-core`, `pybind11`, `cmake` и `ninja` в списке нет.
Путь `_core` ведёт в `site-packages` нового env, а не в исходники.


# Свои тесты
Три теста в `student_tests/test_student.py`:
1. Численный. Массив из десяти чисел, ожидаемые значения считаются обычным
   Python-выражением по спискам, без NumPy. Числа точно представимы в double.
2. Негативный. Четыре и одна ось, `complex128`, `float16`, `None` и строка.
3. На размещение в памяти. `transpose(2, 0, 1)` даёт правильный dtype, ранг и форму, но
   ломает C-порядок.


# Какую ошибку поймал тест
Проверял первый тест так: заменил в ядре `a[i] * b[i] + c[i]` на `a[i] * (b[i] + c[i])`.
Это лишние скобки, код компилируется, форма и dtype результата остаются правильными.
После пересборки и переустановки wheel упало 6 тестов, в том числе мой:

```
FAILED student_tests/test_student.py::test_hand_written_values_on_a_long_last_axis
E       Not equal to tolerance rtol=1e-12, atol=1e-12
E       Mismatched elements: 10 / 10 (100%)
E        [0]: 3.75 (ACTUAL), 3.5 (DESIRED)
E        [1]: -6.75 (ACTUAL), -10.0 (DESIRED)
6 failed, 37 passed
```

# Версии
Ubuntu 22.04.5 LTS в WSL2, x86_64. Python 3.13.15, g++ 11.4.0, C++17.
pybind11 3.1.0, scikit-build-core 1.0.3, CMake 4.4.3, Ninja 1.13.2, NumPy 2.5.3,
build 1.6.1, pytest 9.1.1. `-march=native` не использовал.
