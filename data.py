from sqlalchemy import create_engine, Column, String, MetaData, Table, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

ENGINE = create_engine('sqlite:///bandcamp_data.db')

Base = declarative_base()
Session = sessionmaker(bind=ENGINE)


class Album(Base):
    __tablename__ = 'albums'

    id = Column(String, primary_key=True, unique=True)
    artist = Column(String)
    title = Column(String)
    tags = Column(String)
    url_title = Column(String)
    store = Column(String)
    url = Column(String)

    def __repr__(self):
        return f"<Album(title={self.title}, artist={self.artist})>"


Album.__table__.create(bind=ENGINE, checkfirst=True)
