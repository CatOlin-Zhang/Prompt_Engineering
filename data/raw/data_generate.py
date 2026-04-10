import os
from pathlib import Path

# 1. 确保目录存在
DATA_DIR = Path("./data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2. 定义内容 (确保包含非停用词的实词)
syllabus_content = """
COMP4146 Prompt Engineering for Generative AI
Grading Breakdown:
Lab assignments: 20 percent
In-class quiz: 10 percent
Group project: 20 percent
Final project: 50 percent
The final project includes a group presentation and a personal report.
"""

policies_content = """
Course Policies:
Lab assignments are due every Monday 13:30.
Late submission without prior approval will not be accepted.
Attendance is recommended for all lab sessions.
"""

reviews_content = """
User review snippets about The Beach:
I enjoy travel stories with moral ambiguity and flawed characters.
I do not like slow starts, but I like high psychological tension.
I prefer vivid setting descriptions over action-heavy writing.
"""

# 3. 写入文件 (覆盖旧文件)
files = {
    "Course_Syllabus.txt": syllabus_content,
    "Course_Policies.txt": policies_content,
    "Book_Reviews.txt": reviews_content
}

for filename, content in files.items():
    file_path = DATA_DIR / filename
    file_path.write_text(content.strip(), encoding="utf-8")
    print(f"✅ 已生成: {file_path} (内容长度: {len(content)})")

print("\n🎉 数据准备完成！请再次运行 main.py")