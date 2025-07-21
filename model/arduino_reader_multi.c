#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <string.h>

#define SERIAL_PORT "/dev/ttyACM0"  // **確認這是正確的 Arduino 端口**
#define BAUDRATE B115200

int fd = -1;

void init_serial() {
    if (fd == -1) {
        fd = open(SERIAL_PORT, O_RDWR | O_NOCTTY);
        if (fd == -1) {
            perror("❌ 錯誤：無法開啟串口");
            return;
        }
        
        struct termios options;
        tcgetattr(fd, &options);
        cfsetispeed(&options, BAUDRATE);
        cfsetospeed(&options, BAUDRATE);
        options.c_cflag |= (CLOCAL | CREAD);
        tcsetattr(fd, TCSANOW, &options);
        tcflush(fd, TCIFLUSH);
    }
}

void get_sensor_data(float *pressure1, float *pressure2, float *pressure3, float *pressure4, float *pressure5, float *pressure6) {
    if (fd == -1) {
        *pressure1 = *pressure2 = *pressure3 = *pressure4 = *pressure5 = *pressure6 = -1;
        return;
    }

    char buffer[128];  // **加大緩衝區，避免溢位**
    int bytes_read = 0, total_bytes = 0;

    while (1) {
        bytes_read = read(fd, buffer + total_bytes, 1);
        if (bytes_read > 0) {
            if (buffer[total_bytes] == '\n') {  // **找到換行符表示一條完整的數據**
                buffer[total_bytes] = '\0';
                break;
            }
            total_bytes++;
            if (total_bytes >= 127) {  // **避免超過緩衝區**
                buffer[127] = '\0';
                break;
            }
        } else {
            *pressure1 = *pressure2 = *pressure3 = *pressure4 = *pressure5 = *pressure6 = -1;
            return;
        }
    }

    // **解析數據**
    int parsed = sscanf(buffer, "%f,%f,%f,%f,%f,%f", 
                        pressure1, pressure2, pressure3,
                        pressure4, pressure5, pressure6);

    if (parsed != 6) {  // **確保正確解析 12 個數據**
        printf("❌ Failed to parse buffer: [%s] (parsed fields = %d)\n", buffer, parsed);
        *pressure1 = *pressure2 = *pressure3 = *pressure4 = *pressure5 = *pressure6 = -1;
    }
}

void close_sensor() {
    if (fd != -1) {
        close(fd);
        fd = -1;
    }
}
