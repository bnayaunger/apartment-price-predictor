# 🏠 Apartment Price Predictor

מערכת מבוססת Flask לחיזוי מחירי שכירות של דירות בישראל, המבוססת על מודל למידת מכונה.

## 📌 תיאור הפרויקט
המערכת מאפשרת למשתמש להזין מאפיינים של דירה בטופס אינטרנטי (כגון מיקום, גודל, קומה, חנייה, מעלית ועוד) ולקבל תחזית למחיר השכירות החודשי.  
הפרויקט כולל ממשק HTML, עיבוד נתונים בפייתון ומודל ElasticNet שאומן על דאטה רלוונטי.

### 🏫 פרויקט קורס
הפרויקט פותח במסגרת קורס "ניתוח נתונים מתקדם בפייתון" באוניברסיטה.  
הוא מהווה את החלק השלישי והאחרון מתוך פרויקט גמר תלת-שלבי:
1. כריית הנתונים מאתר אינטרנט (Web Scraping)
2. עיבוד הנתונים ובניית מודלים לחיזוי
3. יצירת אפליקציית Flask להצגת התחזיות

## 🔧 טכנולוגיות וכלים
- Python (pandas, scikit-learn, joblib)
- Flask
- HTML + JavaScript (כולל ולידציה)
- VS Code
- Git + GitHub

## 🚀 איך להפעיל
1. ודא ש־Python מותקן.
2. התקן את הדרישות :
   ```bash
   pip install -r requirements.txt
   ```
3. הפעל את השרת:
   ```bash
   python api.py
   ```
4. עבור לדפדפן וגש לכתובת:
   ```
   http://127.0.0.1:5000/
   ```
5. מלא את הפרטים וקבל חיזוי למחיר הדירה שלך
## 💡 הערות
- המודל אינו מיועד לשימוש מסחרי.
- תהליך עיבוד הנתונים כולל ניקוי, השלמת ערכים חסרים והמרת משתנים קטגוריאליים.
