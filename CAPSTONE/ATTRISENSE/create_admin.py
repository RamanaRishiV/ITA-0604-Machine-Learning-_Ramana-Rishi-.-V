
from getpass import getpass

from app import app
from extensions import db
from database.models import User


with app.app_context():

    name = input("Enter admin name: ").strip()
    email = input("Enter admin email: ").strip().lower()

    if not name or not email:
        print("Name and email are required.")
        raise SystemExit

    user = User.query.filter_by(email=email).first()

    if user:
        print("An account with this email already exists.")
        confirm = input(
            "Promote this account to admin? (yes/no): "
        ).strip().lower()

        if confirm != "yes":
            print("Admin setup cancelled.")
            raise SystemExit

        user.role = "admin"

    else:
        password = getpass("Create admin password: ")
        confirm_password = getpass("Confirm admin password: ")

        if len(password) < 8:
            print("Password must contain at least 8 characters.")
            raise SystemExit

        if password != confirm_password:
            print("Passwords do not match.")
            raise SystemExit

        user = User(
            name=name,
            email=email,
            role="admin"
        )

        user.set_password(password)
        db.session.add(user)

    db.session.commit()

    print("Admin account successfully configured.")
