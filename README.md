# rag_building_documentation
Ai Talent Hub hackaton  
Чтобы поднять проект  
1. ```cp .env.example .env```
2. ```docker compose up```  
Ручки доступны по адресу ```http://localhost:8001/docs```  
```/upload``` принимает на вход файл, обрабатывает и добавляет в бд  
```/upload_folder``` принимает zip архив с файлами, обрабатывает и добавляет в бд  
```/search``` принимает на вход строку запроса и количество записей, которые нужно найти  
chroma доступна по адресу ```http://localhost:8000```