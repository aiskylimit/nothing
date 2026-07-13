#!/bin/bash


CHANGED_DIRS=$(git diff --name-only ORIG_HEAD HEAD | awk -F/ '{print $1}' | sort -u)

if [ -z "$CHANGED_DIRS" ]; then
    echo "Code đã được cập nhật nhưng không có thay đổi nào trong các thư mục con."
    exit 0
fi

echo "Các thư mục có thay đổi: $CHANGED_DIRS"

# Lặp qua các thư mục bị thay đổi và thực thi script
for DIR in $CHANGED_DIRS; do
    if [ -d "$DIR" ] && [ -f "$DIR/run.bash" ]; then
        echo "========================================"
        echo " 🚀 Phát hiện code mới tại: $DIR"
        echo " Đang chạy $DIR/run.bash..."
        echo "========================================"
        
        (
            cd "$DIR" || exit 1
            bash run.bash
        )
        
        if [ $? -ne 0 ]; then
            echo "❌ Lỗi: Xử lý $DIR thất bại!"
            exit 1
        fi
        
        echo "✅ Xử lý xong: $DIR"
    else
        echo "⚠️ Bỏ qua $DIR (Không phải thư mục hoặc không có file run.bash)"
    fi
done

echo "🎉 Hoàn tất toàn bộ quá trình cập nhật!"