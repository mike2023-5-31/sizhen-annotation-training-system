"""本地管理员引导/恢复：python -m app.accounts create NAME --role admin。"""
import argparse
import getpass

from app.services import auth_service as service


def main():
    parser = argparse.ArgumentParser(description="本地账号管理；先运行 python -m app.migrations")
    parser.add_argument("action", choices=["create", "reset-password"])
    parser.add_argument("username")
    parser.add_argument("--role", choices=["admin", "member"], default="member")
    args = parser.parse_args()
    password = getpass.getpass("密码（12–128字符）: ")
    if password != getpass.getpass("重复密码: "):
        parser.error("两次密码不一致")
    try:
        if args.action == "create":
            user = service.create_user(args.username, password, args.role)
        else:
            user = next((u for u in service.list_users() if u["username"] == args.username.lower()), None)
            if not user:
                parser.error("账号不存在")
            user = service.update_user(user["id"], password=password)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"已更新账号 {user['username']} ({user['role']})")


if __name__ == "__main__":
    main()
