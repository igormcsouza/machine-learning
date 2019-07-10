window.onload = function(){
    var post = new Vue({
        el: '#post',

        data: {
            // Post Arguments
            /* Neste caso, essas variáveis devem receber via requisição do backend,
            como eu ainda não aprendi a fazer isso, fica assim num post inicial, mas
            elas devem ser alteradas de acordo com a requisição. Ou seja, eu vou rece-
            ber do front o post que eu quero abrir, e alguma magica deve acontecer aqui
            e puxar do banco de dados o post.
            */
            postTitle: "Em Breve...",
            postSubtitle: "Estamos trabalhando no melhor para o usuário",
            user: "Igor Souza",
            when: "01 de janeiro de 2019",
            
            // Deve haver uma variável para o post, mas como ele tem tags html, o post
            // deve ser um pouco mais complexo do que strings... :/
        },

        methods: {

        }
    })
}
