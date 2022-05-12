from sqlalchemy import create_engine, Column, String, DateTime
import configparser
import datetime
from sqlalchemy.orm import declarative_base, sessionmaker

config = configparser.ConfigParser()
config.read('config.ini')
dbconf = config["DATABASE"]
uname = dbconf['UserName']
pword = dbconf['Password']
addrs = dbconf['Address']
dname = dbconf['Database']
connstring = f'mysql+pymysql://{uname}:{pword}@{addrs}/{dname}?charset=utf8mb4'
ENGINE = create_engine(connstring)

Base = declarative_base()
Session = sessionmaker(bind=ENGINE)


class Album(Base):
    __tablename__ = 'albums'

    id = Column(String(255), primary_key=True, unique=True)
    artist = Column(String(2048))
    title = Column(String(2048))
    tags = Column(String(1024))
    url_title = Column(String(255))
    store = Column(String(255))
    url = Column(String(255))

    def __repr__(self):
        return f"<Album(title={self.title}, artist={self.artist})>"


class Store(Base):
    __tablename__ = 'stores'

    store_name = Column(String(255), primary_key=True, unique=True)
    created_date = Column(DateTime, default=datetime.datetime.utcnow())


Album.__table__.create(bind=ENGINE, checkfirst=True)
Store.__table__.create(bind=ENGINE, checkfirst=True)
