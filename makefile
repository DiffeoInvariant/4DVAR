CC=clang
CFLAGS=-std=c11 -O3 -march=native -mtune=native -fopenmp -shared -fPIC

PETSC_DIR=/usr/local/petsc
PETSC_ARCH=arch-linux2-c-debug
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules
INCLS=-I include/ $(PETSC_CC_INCLUDES) 

TARGET=lib/lib4dvar.so

.PHONY: all clean $(TARGET)

all: $(TARGET)

$(TARGET): src/var.c src/obs.c
	$(CC) $(CFLAGS) $^ -o $@ $(INCLS) $(PETSC_WITH_EXTERNAL_LIB)

test: tests/test_petsc_var.c
	$(CC) -std=c11 -O3 -march=native -mtune=native $^ -o tests/test_var $(INCLS) $(PETSC_WITH_EXTERNAL_LIB) -Llib/ -l4dvar

clean:
	@$(RM) *.o
