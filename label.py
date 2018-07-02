import pandas as pd
from datetime import datetime

df = pd.read_csv('games_form.csv')

total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
missing_gun_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_gun_data.head(20))


# dropping not important pd
df.drop(['Timestamp', 'עמישם התצפיתן'], axis=1, inplace=True)
# clean empty rows
df.dropna(how='all',axis=0)


# fixing the time of the game for students the wrote it wrong, changing minutes to hours
df['שעת המשחק'] = df.loc[:,'שעת המשחק'].map(lambda x: x[x.find(':')+1:]+":00" if x.startswith("0") else x)


# fixing the time they reach the stadium to be from the start of the game
def time_fix(x):
    x1=datetime.strptime(str(x[1]), '%H:%M:%S')
    x0=datetime.strptime(str(x[0]), '%H:%M:%S')
    if (x1 > x0):
        return x1-x0
    return x1-datetime.strptime("00:00:00", '%H:%M:%S')


df.loc[:, 'זמן הגעה למשחק'] = df.loc[:, ['שעת המשחק', 'זמן הגעה למשחק' ]].apply(time_fix,axis=1)

#  mapping the level of woman in the stadium to numbers
df.loc[:, 'נוכחות נשים ביציע'] = df.loc[:, 'נוכחות נשים ביציע'].map({'נמוכה':1,'בינונית':2,'גבוהה':3})
df.loc[:, 'נוכחות נשים ביציע']= df.loc[:, 'נוכחות נשים ביציע'].fillna(0) # fill missing with 0

#  mapping the level of kids in the stadium to numbers
df.loc[:, 'נוכחות ילדים (עד גיל 10) ביציע'] = df.loc[:, 'נוכחות ילדים (עד גיל 10) ביציע'].map({'נמוכה':1,'בינונית':2, 'גבוהה':3})
df.loc[:, 'נוכחות ילדים (עד גיל 10) ביציע'] = df.loc[:, 'נוכחות ילדים (עד גיל 10) ביציע'].fillna(0)


# mapping over 4 to 5, and nan to 0
df.loc[:,'נוכחות שחקנים ערבים בהרכב הקבוצה המארחת והקבוצה האורחת (שחקנים יהודים במשחקי בני סכנין)']=df.loc[:,'נוכחות שחקנים ערבים בהרכב הקבוצה המארחת והקבוצה האורחת (שחקנים יהודים במשחקי בני סכנין)'].map({"מעל 4":5})
df.loc[:,'נוכחות שחקנים ערבים בהרכב הקבוצה המארחת והקבוצה האורחת (שחקנים יהודים במשחקי בני סכנין)']=df.loc[:,'נוכחות שחקנים ערבים בהרכב הקבוצה המארחת והקבוצה האורחת (שחקנים יהודים במשחקי בני סכנין)'].fillna(0)

# mapping over 4 to 5, and nan to 0
df.loc[:,'נוכחות שחקנים כהי עור בהרכב הקבוצה המארחת והקבוצה האורחת']=df.loc[:,'נוכחות שחקנים כהי עור בהרכב הקבוצה המארחת והקבוצה האורחת'].map({"מעל 4":5})
df.loc[:,'נוכחות שחקנים כהי עור בהרכב הקבוצה המארחת והקבוצה האורחת']=df.loc[:,'נוכחות שחקנים כהי עור בהרכב הקבוצה המארחת והקבוצה האורחת'].fillna(0)


# remove any description of the event
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.str.startswith('אירוע')]
df = df.loc[:, ~df.columns.str.startswith('מה סיבת האירוע')]

#df.loc[:,df.columns.str.startswith('דקה')] = df.loc[:,df.columns.str.startswith('דקה')].apply(lambda x: pd.Series(map(lambda y: float(y.split("-")[1])-float(y.split("-")[0]),x)).sum())


df.to_csv("game_fix.csv", sep=',', encoding='utf-8')
