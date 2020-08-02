import smtplib


def sendmail(to_email, message, Cofirmation_mail):

    from_email = "mail.developer.mandar@gmail.com"

    response = {}
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(from_email, "Mandar@18")
            print("Sending Mail:", to_email)
            s.sendmail(from_email, to_email, message)
        response['email_status'] = "Success"
    except Exception as err:
        print(err)
        response['email_status'] = "Failed"

    return response
# if __name__ == "__main__":
# sendmail("harshpatel682@gmail.com", "Hello", "Paid paisa")