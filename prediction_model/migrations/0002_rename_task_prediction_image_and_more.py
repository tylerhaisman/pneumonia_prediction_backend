# Generated by Django 5.0.4 on 2024-05-04 00:02

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('prediction_model', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='prediction',
            old_name='task',
            new_name='image',
        ),
        migrations.RemoveField(
            model_name='prediction',
            name='completed',
        ),
        migrations.RemoveField(
            model_name='prediction',
            name='updated',
        ),
        migrations.RemoveField(
            model_name='prediction',
            name='user',
        ),
    ]
