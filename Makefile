COMPILER  = g++
INCLUDES = -I/usr/local/include/opencv4/
main: main.cpp
	$(COMPILER) -std=c++17 main.cpp `pkg-config --cflags opencv4` `pkg-config --libs opencv4` -o main 

clean:
	rm -f *.o main
