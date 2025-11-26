# ==========================================
# 编译器设置
# ==========================================
CC = gcc
CFLAGS = -Iinclude -Isrc -O3 -mavx2 -mfma -Wall -Wextra
LDFLAGS = -lm

# ==========================================
# 平台检测与命令适配 (新增部分)
# ==========================================
# 判断是否为 Windows
ifeq ($(OS),Windows_NT)
    # Windows 下的删除命令
    RM = del /Q /F
    # Windows 下的可执行文件后缀
    TARGET_EXT = .exe
    # 路径转换函数：把 / 替换为 \ (Windows cmd 需要反斜杠)
    FixPath = $(subst /,\,$1)
else
    # Linux/Mac 下的删除命令
    RM = rm -f
    # Linux 下没有后缀
    TARGET_EXT = 
    # 路径不需要转换
    FixPath = $1
endif

# ==========================================
# 文件与目标
# ==========================================
TARGET = fft_test

# 默认使用Perform.c作为主文件
MAIN_FILE ?= Perform.c

# 根据MAIN_FILE变量构建源文件列表
SRCS = $(MAIN_FILE) src/fft_tables.c src/fft_avx_float.c src/fft_avx_fixed.c src/fft_utils.c
OBJS = $(SRCS:.c=.o)

# ==========================================
# 编译规则
# ==========================================

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	$(CC) $(OBJS) -o $(TARGET)$(TARGET_EXT) $(LDFLAGS)
	@echo "Build successful! Run with ./$(TARGET)$(TARGET_EXT)"

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ==========================================
# 清理规则 (删除所有.o和.exe文件)
# ==========================================
clean:
	@echo "Cleaning up all object and executable files..."
	$(RM) $(call FixPath,*.o) $(call FixPath,*.exe) $(call FixPath,src\*.o) $(call FixPath,*.obj) $(call FixPath,src\*.obj)
	@echo "Clean up complete."