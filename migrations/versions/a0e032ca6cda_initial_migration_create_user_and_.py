"""Initial migration: Create User and PredictionHistory tables

Revision ID: a0e032ca6cda
Revises: 
Create Date: 2025-06-23 22:18:31.320693

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'a0e032ca6cda'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('name', sa.String(length=120), nullable=False))
        batch_op.add_column(sa.Column('created_at', sa.DateTime(), nullable=True))
        batch_op.alter_column('username',
               existing_type=mysql.VARCHAR(length=80),
               type_=sa.String(length=120),
               existing_nullable=False)
        batch_op.alter_column('password_hash',
               existing_type=mysql.VARCHAR(length=128),
               type_=sa.String(length=255),
               existing_nullable=False)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column('password_hash',
               existing_type=sa.String(length=255),
               type_=mysql.VARCHAR(length=128),
               existing_nullable=False)
        batch_op.alter_column('username',
               existing_type=sa.String(length=120),
               type_=mysql.VARCHAR(length=80),
               existing_nullable=False)
        batch_op.drop_column('created_at')
        batch_op.drop_column('name')

    # ### end Alembic commands ###
