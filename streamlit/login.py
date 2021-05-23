import streamlit as st
from google.cloud import firestore
def loginPage():

    st.header("Lütfen Giriş Yapınız")
    mailLogin = st.text_input("E-mail")
    passwdLogin = st.text_input("Şifre", type="password")
    login = st.button("Giriş")
    if login:
        if (mailLogin == "") or (passwdLogin == "") :
            st.error("Alanlar Boş olamaz!")
        else:
            # Authenticate to Firestore with the JSON account key.
            db = firestore.Client.from_service_account_json("firestore-key.json")

            # Create a reference to the Google post.
            doc_ref = db.collection("users")

            # Then get the data at that reference.
            doc = doc_ref.get() 

            for i in range(1,len(doc)+1):
                temp = db.collection("users").document(str(i))
                temp = temp.get()
                datas = temp.to_dict()
                if (datas["mail"] == mailLogin) and (datas["pass"] == passwdLogin):
                    st.success("Giriş işleminiz başarılı! Anasayfaya gidebilirsiniz.")
                    return True
            st.error("Kayıt Bulunamadı")
    

    st.header("HESABINIZ YOK MU? HEMEN KAYDOLUN!")
    name = st.text_input("İsminiz")
    surname = st.text_input("Soy İsminiz")
    mailSignUp = st.text_input("Mail Adresiniz?")
    passwdSignUp = st.text_input("Şifreniz", type="password")
    signup = st.button("Kayıt Ol")
    if signup:
        if (name == "") or (surname == "") or (mailSignUp == "") or (passwdSignUp == "") :
            st.error("Hiçbir Alan Boş Kalamaz!")
        else:
            # Authenticate to Firestore with the JSON account key.
            db = firestore.Client.from_service_account_json("firestore-key.json")

            # Create a reference to the Google post.
            doc_ref = db.collection("users")

            # Then get the data at that reference.
            doc = doc_ref.get() 
            lens = len(doc)
            doc_ref.add(document_data= {
                "name": name,
                "surname": surname,
                "mail": mailSignUp,
                "pass": passwdSignUp
            }, document_id= str(lens+1))
            st.success("Başarılı bir şekilde kaydoldunuz!")
            return False
    return False
if __name__ == "__main__":
    loginPage()










